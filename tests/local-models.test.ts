import assert from "node:assert/strict";
import { describe, it, beforeEach } from "node:test";
import {
	normalizeOllamaRootUrl,
	normalizeOpenAIBaseUrl,
	sanitizeUserSettings,
} from "../src/core/settings.ts";
import {
	clearOllamaShowCache,
	ollamaShowCache,
	probeOllama,
	probeVLLM,
} from "../src/server/local-models.ts";
import {
	clearLocalProvider,
	createLocalProvider,
	piModels,
	setLocalProvider,
	toPiModelDefinition,
} from "../src/agent/models.ts";
import { parseLocalProvidersSettingsBody } from "../src/server/settings-body.ts";
import { resolveKeyStatuses } from "../src/server/keys.ts";
import type { ModelCatalogEntry } from "../src/core/contracts/models.ts";

describe("local models URL normalization", () => {
	it("normalizeOllamaRootUrl strips trailing slashes and /v1", () => {
		assert.equal(normalizeOllamaRootUrl("http://localhost:11434"), "http://localhost:11434");
		assert.equal(normalizeOllamaRootUrl("http://localhost:11434/"), "http://localhost:11434");
		assert.equal(normalizeOllamaRootUrl("http://localhost:11434/v1"), "http://localhost:11434");
		assert.equal(normalizeOllamaRootUrl("http://localhost:11434/v1/"), "http://localhost:11434");
	});

	it("normalizeOpenAIBaseUrl ensures trailing /v1", () => {
		assert.equal(normalizeOpenAIBaseUrl("http://localhost:8000"), "http://localhost:8000/v1");
		assert.equal(normalizeOpenAIBaseUrl("http://localhost:8000/"), "http://localhost:8000/v1");
		assert.equal(normalizeOpenAIBaseUrl("http://localhost:8000/v1"), "http://localhost:8000/v1");
		assert.equal(normalizeOpenAIBaseUrl("http://localhost:8000/v1/"), "http://localhost:8000/v1");
	});
});

describe("probeOllama discovery", () => {
	beforeEach(() => {
		clearOllamaShowCache();
	});

	it("filters embedding-only models and inspects chat model capabilities", async () => {
		let showCalls = 0;
		const mockFetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
			const url = String(input);
			if (url.endsWith("/api/tags")) {
				return new Response(
					JSON.stringify({
						models: [
							{ name: "qwen2.5-coder:14b", digest: "sha256:11111" },
							{ name: "nomic-embed-text:latest", digest: "sha256:22222" },
							{ name: "llava:7b", digest: "sha256:33333" },
						],
					}),
					{ status: 200, headers: { "Content-Type": "application/json" } },
				);
			}
			if (url.endsWith("/api/show")) {
				showCalls++;
				const body = JSON.parse(String(init?.body)) as { name: string };
				if (body.name.includes("nomic-embed")) {
					return new Response(
						JSON.stringify({
							capabilities: ["embedding"],
							details: { family: "nomic-bert" },
						}),
						{ status: 200, headers: { "Content-Type": "application/json" } },
					);
				}
				if (body.name.includes("llava")) {
					return new Response(
						JSON.stringify({
							capabilities: ["completion", "vision"],
							parameters: "num_ctx 8192",
						}),
						{ status: 200, headers: { "Content-Type": "application/json" } },
					);
				}
				return new Response(
					JSON.stringify({
						capabilities: ["completion", "tools"],
						parameters: "num_ctx 16384",
						model_info: { "qwen2.context_length": 32768 },
					}),
					{ status: 200, headers: { "Content-Type": "application/json" } },
				);
			}
			return new Response("Not found", { status: 404 });
		};

		const res = await probeOllama("http://localhost:11434/v1", 32768, mockFetch as typeof fetch);
		assert.ok(res.ok);
		assert.equal(res.models.length, 2, "nomic-embed-text should be filtered out");

		const qwen = res.models.find((m) => m.id === "qwen2.5-coder:14b");
		assert.ok(qwen);
		assert.equal(qwen.provider, "ollama");
		assert.equal(qwen.contextWindow, 16384, "parameters num_ctx takes precedence");
		assert.equal(qwen.maxTokens, 4096);
		assert.deepEqual(qwen.input, ["text"]);
		assert.deepEqual(qwen.cost, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });

		const llava = res.models.find((m) => m.id === "llava:7b");
		assert.ok(llava);
		assert.deepEqual(llava.input, ["text", "image"], "vision capability detected");
		assert.equal(llava.contextWindow, 8192);

		// Subsequent call should hit digest cache without additional /api/show calls
		const initialShowCalls = showCalls;
		const res2 = await probeOllama("http://localhost:11434", 32768, mockFetch as typeof fetch);
		assert.ok(res2.ok);
		assert.equal(showCalls, initialShowCalls, "cached models should not re-query /api/show");
	});
});

describe("probeVLLM discovery", () => {
	it("extracts max_model_len and sends bearer token", async () => {
		let authHeaderReceived: string | null = null;
		const mockFetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
			const headers = init?.headers as Record<string, string> | undefined;
			authHeaderReceived = headers?.Authorization ?? null;

			return new Response(
				JSON.stringify({
					data: [
						{ id: "Qwen/Qwen2.5-Coder-32B-Instruct", max_model_len: 65536 },
						{ id: "mistralai/Mistral-7B-Instruct-v0.3", context_length: 32768 },
					],
				}),
				{ status: 200, headers: { "Content-Type": "application/json" } },
			);
		};

		const res = await probeVLLM("http://localhost:8000", "my-secret-token", 32768, mockFetch as typeof fetch);
		assert.ok(res.ok);
		assert.equal(authHeaderReceived, "Bearer my-secret-token");
		assert.equal(res.models.length, 2);

		const qwen = res.models.find((m) => m.id === "Qwen/Qwen2.5-Coder-32B-Instruct");
		assert.ok(qwen);
		assert.equal(qwen.provider, "custom");
		assert.equal(qwen.contextWindow, 65536);
		assert.equal(qwen.cost.input, 0);

		const mistral = res.models.find((m) => m.id === "mistralai/Mistral-7B-Instruct-v0.3");
		assert.ok(mistral);
		assert.equal(mistral.contextWindow, 32768);
	});

	it("detects vision capability on models like minicpm-v, qwen2-vl, and pixtral", async () => {
		const mockFetch = async (): Promise<Response> => {
			return new Response(
				JSON.stringify({
					data: [
						{ id: "openbmb/MiniCPM-V-2_6", max_model_len: 4096 },
						{ id: "mistralai/Pixtral-12B-2409", max_model_len: 16384 },
						{ id: "THUDM/glm-4v-9b", max_model_len: 8192 },
					],
				}),
				{ status: 200, headers: { "Content-Type": "application/json" } },
			);
		};

		const res = await probeVLLM("http://localhost:8000/v1", undefined, 32768, mockFetch as typeof fetch);
		assert.ok(res.ok);
		assert.equal(res.models.length, 3);
		assert.ok(res.models.every((m) => m.input.includes("image")), "all should be detected as vision capable");
	});
});

describe("createLocalProvider and compatibility flags", () => {
	it("configures ApiKeyAuth contract and strict compatibility flags", async () => {
		const entry: ModelCatalogEntry = {
			id: "qwen2.5:14b",
			name: "Ollama / qwen2.5:14b",
			provider: "ollama",
			reasoning: false,
			input: ["text"],
			supportedLevels: ["off"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 4096,
			maxTokens: 2048,
		};

		const piModel = toPiModelDefinition(entry, "http://localhost:11434/v1");
		assert.equal(piModel.api, "openai-completions");
		assert.equal(piModel.baseUrl, "http://localhost:11434/v1");
		assert.equal(piModel.compat?.supportsStore, false);
		assert.equal(piModel.compat?.supportsDeveloperRole, false);
		assert.equal(piModel.compat?.supportsStrictMode, false);
		assert.equal(piModel.compat?.maxTokensField, "max_tokens");
		assert.equal(piModel.compat?.supportsReasoningEffort, false);

		// Tight VRAM maxTokens clamp check
		const tightEntry: ModelCatalogEntry = {
			...entry,
			contextWindow: 2048,
			maxTokens: 0,
		};
		const clampedModel = toPiModelDefinition(tightEntry, "http://localhost:11434/v1");
		assert.equal(clampedModel.maxTokens, 1024, "clamped to half of context window");

		// Provider creation & ApiKeyAuth structure
		const provider = createLocalProvider("ollama", "http://localhost:11434", undefined, [entry]);
		assert.equal(provider.id, "ollama");
		assert.ok(provider.auth.apiKey);
		assert.equal(provider.auth.apiKey.name, "Ollama");

		// Test resolve contract
		const resolved = await provider.auth.apiKey.resolve({} as never);
		assert.equal(resolved?.auth.apiKey, "ollama");
		assert.equal(resolved?.source, "local");

		// Register in piModels collection and verify retrieval
		setLocalProvider(provider);
		const models = piModels();
		const retrieved = models.getModel("ollama", "qwen2.5:14b");
		assert.ok(retrieved);
		assert.equal(retrieved.id, "qwen2.5:14b");

		// De-register on disable
		clearLocalProvider("ollama");
		const cleared = models.getModel("ollama", "qwen2.5:14b");
		assert.equal(cleared, undefined, "model must be removed after clear");
	});
});

describe("settings parsing and sanitization", () => {
	it("parseLocalProvidersSettingsBody validates valid config", () => {
		const parsed = parseLocalProvidersSettingsBody({
			ollama: { enabled: true, baseUrl: "http://127.0.0.1:11434" },
			custom: { enabled: false, baseUrl: "http://127.0.0.1:8000/v1", defaultContextWindow: 65536 },
		});
		assert.ok(parsed.ok);
		if (parsed.ok) {
			assert.equal(parsed.settings.ollama?.enabled, true);
			assert.equal(parsed.settings.ollama?.baseUrl, "http://127.0.0.1:11434");
			assert.equal(parsed.settings.custom?.defaultContextWindow, 65536);
		}
	});

	it("sanitizeUserSettings preserves and clamps localProviders delta", () => {
		const sanitized = sanitizeUserSettings({
			version: 1,
			localProviders: {
				ollama: { enabled: true, baseUrl: "  http://localhost:11434  " },
				custom: { defaultContextWindow: -5 }, // invalid positive number
			},
		});
		assert.ok(sanitized);
		assert.equal(sanitized?.localProviders?.ollama?.enabled, true);
		assert.equal(sanitized?.localProviders?.ollama?.baseUrl, "http://localhost:11434");
		assert.equal(sanitized?.localProviders?.custom?.defaultContextWindow, undefined);
	});
});

describe("LOCAL_AI_API_KEY resolution", () => {
	it("resolveKeyStatuses includes LOCAL_AI_API_KEY when present in file or env", () => {
		const statuses = resolveKeyStatuses(
			{ LOCAL_AI_API_KEY: "secret-token-12345" },
			{},
			new Set(),
		);
		const localKey = statuses.find((s) => s.id === "LOCAL_AI_API_KEY");
		assert.ok(localKey);
		assert.equal(localKey.present, true);
		assert.equal(localKey.source, "file");
	});
});
