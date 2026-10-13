// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Globalization;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Sessions;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Knowledge;
using TensorSharp.Server.Knowledge;

namespace TensorAgent.Core.Hosting;

public static partial class WebUiRoutes
{
    /// <summary>Copies explicitly selected saved chats and their local files into the durable knowledge library.</summary>
    public static void MapKnowledgeChats(this LoopbackServer server, KnowledgeStore knowledge,
        ConversationStore conversations, string uploadDirectory, CodeArtifactStore? artifacts = null)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(knowledge);
        ArgumentNullException.ThrowIfNull(conversations);
        ArgumentException.ThrowIfNullOrWhiteSpace(uploadDirectory);
        uploadDirectory = Path.GetFullPath(uploadDirectory);

        server.MapGet("/api/knowledge/chats", (request, _) => KnowledgeRoute(() =>
        {
            string query = request.Query("q")?.Trim() ?? "";
            if (query.Length > 2000) throw new ArgumentException("Chat search exceeds 2000 characters.");
            int offset = QueryInteger(request, "offset", 0, 0, int.MaxValue);
            int limit = QueryInteger(request, "limit", 100, 1, 200);
            var all = conversations.List().Where(chat => query.Length == 0
                || chat.Title.Contains(query, StringComparison.OrdinalIgnoreCase)).ToArray();
            var chats = all.Skip(offset).Take(limit).ToArray();
            return Task.FromResult<LoopbackResponse?>(KnowledgeJson(new { chats, total = all.Length,
                nextOffset = offset < all.Length - chats.Length ? offset + chats.Length : (int?)null }));
        }));

        server.MapPost("/api/knowledge/collections/{collection}/chats", (request, ct) => KnowledgeRoute(async () =>
        {
            string collection = request.RouteValues["collection"];
            _ = knowledge.ListDocuments(collection);
            JsonElement body = await ReadKnowledgeJson(request, ct);
            if (!body.TryGetProperty("conversationIds", out var selected) || selected.ValueKind != JsonValueKind.Array
                || selected.GetArrayLength() is < 1 or > 32)
                throw new ArgumentException("conversationIds must be an array of 1–32 saved chat IDs.");
            string[] ids = selected.EnumerateArray().Select(value => value.ValueKind == JsonValueKind.String
                && ConversationStore.IsValidId(value.GetString()) ? value.GetString()!
                : throw new ArgumentException("Saved chat IDs must be 32 hexadecimal characters.")).Distinct(StringComparer.Ordinal).ToArray();

            var documents = new List<KnowledgeDocument>();
            var importedChats = new List<object>();
            var errors = new List<object>();
            foreach (string id in ids)
            {
                ct.ThrowIfCancellationRequested();
                Conversation? conversation = null;
                try
                {
                    conversation = conversations.Load(id) ?? throw new KeyNotFoundException("Saved chat not found or unreadable.");
                    if (conversation.IsEmpty) throw new ArgumentException("Empty chats cannot be added to knowledge.");
                    string prefix = "Chats/" + id + "/";
                    string transcript = KnowledgeChatTranscript(conversation, knowledge.Options.MaxExtractedCharacters);
                    await using var text = new MemoryStream(Encoding.UTF8.GetBytes(transcript));
                    KnowledgeDocument document = await knowledge.ImportAsync(collection, "conversation.md", text,
                        "text/markdown; charset=utf-8", prefix + "conversation.md", ct).ConfigureAwait(false);
                    documents.Add(document);

                    int attachmentCount = 0;
                    foreach (var file in KnowledgeChatFiles(conversation))
                    {
                        ct.ThrowIfCancellationRequested();
                        try
                        {
                            (string fullPath, string relativePath) = KnowledgeChatFile(file, uploadDirectory, artifacts);
                            string name = file.DisplayName ?? Path.GetFileName(fullPath);
                            if (string.IsNullOrWhiteSpace(name) || name.Length > 512 || name is "." or ".."
                                || name.Any(c => char.IsControl(c) || c is '/' or '\\'))
                                name = Path.GetFileName(fullPath);
                            string contentType = ContentTypes.For(fullPath);
                            name = UploadNaming.ResolveFileName(name, contentType, fullPath);
                            string storedExtension = Path.GetExtension(fullPath);
                            if ((contentType.StartsWith("image/", StringComparison.Ordinal)
                                    || contentType.StartsWith("audio/", StringComparison.Ordinal)
                                    || contentType.StartsWith("video/", StringComparison.Ordinal))
                                && !Path.GetExtension(name).Equals(storedExtension, StringComparison.OrdinalIgnoreCase))
                                name = Path.GetFileNameWithoutExtension(name) + storedExtension;
                            if (name.Length > 512) name = Path.GetFileName(fullPath);
                            // The stored upload name (or artifact path) distinguishes two attachments
                            // sharing a display name, and remains stable when this chat is imported again.
                            await using var source = File.OpenRead(fullPath);
                            documents.Add(await knowledge.ImportAsync(collection, name, source, contentType,
                                prefix + relativePath + "/" + name, ct).ConfigureAwait(false));
                            attachmentCount++;
                        }
                        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
                        catch (PdfOcrSetupException)
                        {
                            errors.Add(new { conversationId = id, title = conversation.Title, fileName = file.DisplayName ?? file.Reference,
                                code = "pdf_ocr_setup_failed", error = "PDF reading tools could not be prepared. Please retry." });
                        }
                        catch (Exception ex) when (KnowledgeChatImportError(ex))
                        {
                            errors.Add(new { conversationId = id, title = conversation.Title, fileName = file.DisplayName ?? file.Reference,
                                error = KnowledgeChatError(ex) });
                        }
                    }
                    importedChats.Add(new { id, title = conversation.Title, documentId = document.Id, attachmentCount });
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
                catch (Exception ex) when (KnowledgeChatImportError(ex))
                {
                    errors.Add(new { conversationId = id, title = conversation?.Title, error = KnowledgeChatError(ex) });
                }
            }
            return KnowledgeJson(new { documents, importedChats, errors });
        }));
    }

    private static bool KnowledgeChatImportError(Exception error) => error is ArgumentException or InvalidDataException
        or KnowledgeLimitException or JsonException or KeyNotFoundException or IOException or UnauthorizedAccessException;

    private static string KnowledgeChatError(Exception error) => error is IOException or UnauthorizedAccessException
        ? "The saved chat or attachment file could not be read. It may have been removed or is unavailable."
        : error.Message;

    private static string KnowledgeChatTranscript(Conversation conversation, int maxCharacters)
    {
        var output = new StringBuilder();
        void Append(string text)
        {
            if (text.Length > maxCharacters - output.Length)
                throw new KnowledgeLimitException("Saved chat exceeds the indexed text limit. Split it into smaller sources.");
            output.Append(text);
        }
        Append("# " + conversation.Title + "\n\n");
        Append("Saved chat: " + conversation.Id + "\nCreated: " + conversation.CreatedAt.ToString("O", CultureInfo.InvariantCulture)
            + "\nUpdated: " + conversation.UpdatedAt.ToString("O", CultureInfo.InvariantCulture) + "\n");
        if (!string.IsNullOrWhiteSpace(conversation.ModelId)) Append("Model: " + conversation.ModelId + "\n");
        int number = 0;
        foreach (var message in conversation.Messages)
        {
            Append("\n## " + (++number).ToString(CultureInfo.InvariantCulture) + ". " + message.Role + "\n\n");
            Append(message.Content);
            Append("\n");
            // Private reasoning and runtime counters are not conversation content.
            // File names give text-only retrieval useful context even when media needs
            // a compatible model before its original can be described or transcribed.
            foreach (var file in KnowledgeChatFiles(new Conversation { Messages = [message] }))
                Append("\nAttachment: " + (file.DisplayName ?? Path.GetFileName(file.Reference)) + "\n");
            if (!string.IsNullOrWhiteSpace(message.ImagePrompt)) Append("\nImage prompt: " + message.ImagePrompt + "\n");
        }
        return output.ToString();
    }

    private sealed record KnowledgeChatAttachment(string Reference, string? DisplayName = null, bool Artifact = false);

    private static IEnumerable<KnowledgeChatAttachment> KnowledgeChatFiles(Conversation conversation)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var message in conversation.Messages)
        {
            var names = new Dictionary<string, string>(StringComparer.Ordinal);
            if (message.Attachments is not null)
                foreach (var attachment in message.Attachments)
                    if (!string.IsNullOrWhiteSpace(attachment.File) && !string.IsNullOrWhiteSpace(attachment.FileName))
                        names[Path.GetFileName(attachment.File)] = attachment.FileName;
            if (message.TextFilePaths is not null && message.TextFileNames is not null)
                for (int index = 0; index < Math.Min(message.TextFilePaths.Count, message.TextFileNames.Count); index++)
                    names[Path.GetFileName(message.TextFilePaths[index])] = message.TextFileNames[index];
            foreach (string reference in message.ReferencedUploads)
                if (!string.IsNullOrWhiteSpace(reference) && seen.Add("upload:" + reference))
                    yield return new(reference, names.GetValueOrDefault(reference));
            if (message.Artifacts is not null)
                foreach (var artifact in message.Artifacts)
                    if (!string.IsNullOrWhiteSpace(artifact.Url) && seen.Add("artifact:" + artifact.Url))
                        yield return new(artifact.Url, artifact.Name, Artifact: true);
        }
    }

    private static (string FullPath, string RelativePath) KnowledgeChatFile(KnowledgeChatAttachment file,
        string uploadDirectory, CodeArtifactStore? artifacts)
    {
        if (!file.Artifact)
        {
            // ReferencedUploads returns bare file names, never a client-selected local path.
            string name = file.Reference;
            if (name.Length > 512 || name is "." or ".." || name.Any(c => char.IsControl(c) || c is '/' or '\\' or ':')
                || !SkillPathGuard.TryResolveExistingFile(uploadDirectory, name, out string? fullPath, out _))
                throw new InvalidDataException("Attachment is missing or is outside the upload library.");
            return (fullPath!, "attachments/" + name);
        }
        const string prefix = "/api/code/artifacts/";
        if (artifacts is null || !file.Reference.StartsWith(prefix, StringComparison.Ordinal)
            || file.Reference.IndexOfAny(['?', '#']) >= 0)
            throw new InvalidDataException("Produced file is unavailable or is outside the artifact library.");
        string[] parts = file.Reference[prefix.Length..].Split('/').Select(Uri.UnescapeDataString).ToArray();
        if (parts.Length < 2 || parts.Any(part => string.IsNullOrWhiteSpace(part) || part is "." or ".."
                || part.Any(c => char.IsControl(c) || c is '/' or '\\' or ':'))
            || !artifacts.TryResolve(parts[0], string.Join('/', parts.Skip(1)), out string? artifactPath, out _))
            throw new InvalidDataException("Produced file is missing or is outside the artifact library.");
        return (artifactPath!, "artifacts/" + string.Join('/', parts));
    }
}
