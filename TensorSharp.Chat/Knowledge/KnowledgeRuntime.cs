using System;
using TensorSharp.Knowledge;

namespace TensorSharp.Server.Knowledge;

/// <summary>Host configuration. TensorAgent can opt into app-private PDF OCR provisioning.</summary>
public static class KnowledgeRuntime
{
    public static KnowledgeStore CreateStore(string root, ModelService modelService = null,
        PdfOcrProvisioner managedPdfOcr = null)
    {
        var options = new KnowledgeOptions();
        GgufKnowledgeEmbeddingProvider embeddings = null;
        try
        {
            string model = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_EMBEDDING_MODEL");
            if (!string.IsNullOrWhiteSpace(model))
                embeddings = new GgufKnowledgeEmbeddingProvider(model,
                    Environment.GetEnvironmentVariable("TS_KNOWLEDGE_EMBEDDING_BACKEND") ?? "CPU",
                    Environment.GetEnvironmentVariable("TS_KNOWLEDGE_QUERY_PREFIX"),
                    Environment.GetEnvironmentVariable("TS_KNOWLEDGE_DOCUMENT_PREFIX"));
            var media = modelService == null ? null : new ModelKnowledgeMediaExtractor(modelService);
            var requested = PdfOcrOptions.FromEnvironment();
            IKnowledgePdfPageExtractor pages = managedPdfOcr != null && PdfOcrProvisioner.ShouldManage(requested)
                ? new ManagedPdfKnowledgePageExtractor(managedPdfOcr.PrepareAsync, requested)
                : new PdfKnowledgePageExtractor(requested);
            IKnowledgeExtractor extractor = new BuiltInKnowledgeExtractor(options, media, pages);
            return new KnowledgeStore(root, extractor, embeddings, options);
        }
        catch { embeddings?.Dispose(); throw; }
    }
}
