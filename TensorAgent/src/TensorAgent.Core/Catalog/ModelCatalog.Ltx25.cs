namespace TensorAgent.Core.Catalog;

public static partial class ModelCatalog
{
    // Exact files used by the native distilled text/image-to-video/audio pipeline.
    // Official companions require approved publisher access on Hugging Face.
    private static CatalogModel Ltx25Distilled() => new()
    {
        Id = "ltx-2.5-distilled-q4km",
        DisplayName = "LTX 2.5 Distilled",
        Family = CatalogFamily.Ltx25,
        Kind = CatalogArchitectureKind.Diffusion,
        // GGUF tensor inventory includes ~21B parameters with its AV connectors;
        // the authentic custom Gemma4 encoder is a separate 12B checkpoint.
        Parameters = "21B + 12B",
        Quantization = "Q4_K_M + BF16",
        Files = new[]
        {
            new CatalogFile(CatalogFileRole.Weights, "LTX-2.5-Distilled-Q4_K_M.gguf",
                "https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF/resolve/7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0/LTX-2.5-Distilled-Q4_K_M.gguf",
                15_687_639_488, "0f51eb0d82b19bddbfb3b0371a65217844ea03750f27dd733528f22152e0e0d0"),
            new CatalogFile(CatalogFileRole.TextEncoder, "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                "https://huggingface.co/Lightricks/LTX-2.5/resolve/49b093a39c19169a8a81721dcc541e07e945232f/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                26_263_858_182, "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1"),
            new CatalogFile(CatalogFileRole.Vae, "ltx-2.5-video-vae-conv-bf16.safetensors",
                "https://huggingface.co/Lightricks/LTX-2.5/resolve/49b093a39c19169a8a81721dcc541e07e945232f/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                1_452_269_922, "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d"),
            new CatalogFile(CatalogFileRole.AudioVae, "ltx-2.5-audio-vae-bf16.safetensors",
                "https://huggingface.co/Lightricks/LTX-2.5/resolve/49b093a39c19169a8a81721dcc541e07e945232f/vae/ltx-2.5-audio-vae-bf16.safetensors",
                364_866_540, "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5"),
            // The publisher's two-stage recipe: half-size pass, x2 latent upscale, refinement.
            new CatalogFile(CatalogFileRole.LatentUpscaler, "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                "https://huggingface.co/Lightricks/LTX-2.5/resolve/49b093a39c19169a8a81721dcc541e07e945232f/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                995_778_752, "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8"),
        },
        Modalities = CatalogModalities.Image | CatalogModalities.VideoOutput | CatalogModalities.AudioOutput,
        // Conservative device tier; initial full-generation validation uses short clips.
        MinDeviceMemoryGB = 48,
        ContextLength = 0,
        KvCacheDtype = "f16",
        Sampling = new CatalogSampling(1f, 0, 1f, 0f),
        LeanCaches = true,
        Experimental = true,
        Notes = "catalog.model.ltx25.notes",
        License = "catalog.license.ltx25",
    };
}
