// Shared bounded OOXML reader; retained as a forwarding adapter for Jev attachments.
using System.Threading;
using TensorSharp.Knowledge;
namespace TensorSharp.Server.Jev;
internal static class JevAttachmentOfficeText
{
    internal const long MaxExpandedBytes = OfficeKnowledgeText.MaxExpandedBytes;
    internal static string Extract(string path, string extension, int maxCharacters, CancellationToken cancellationToken)
        => OfficeKnowledgeText.Extract(path, extension, maxCharacters, cancellationToken);
}
