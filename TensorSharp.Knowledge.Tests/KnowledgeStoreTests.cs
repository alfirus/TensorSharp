using System.Text;
using System.Text.Json.Nodes;

namespace TensorSharp.Knowledge.Tests;

public sealed class KnowledgeStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensorsharp-knowledge-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { if (Directory.Exists(_root)) Directory.Delete(_root, true); }

    [Fact]
    public async Task FolderPathsPreserveDuplicateBasenamesAndReplacementIdentityAcrossRestart()
    {
        string collectionId, firstId, secondId;
        using (var store = new KnowledgeStore(_root, new TextExtractor()))
        {
            collectionId = (await store.CreateCollectionAsync("Project" )).Id;
            using var firstBytes = new MemoryStream(Encoding.UTF8.GetBytes("orchard north"));
            var first = await store.ImportAsync(collectionId, "notes.txt", firstBytes, "text/plain", "Project/north/notes.txt");
            using var secondBytes = new MemoryStream(Encoding.UTF8.GetBytes("orchard south"));
            var second = await store.ImportAsync(collectionId, "notes.txt", secondBytes, "text/plain", "Project/south/notes.txt");
            firstId = first.Id; secondId = second.Id;
            Assert.NotEqual(firstId, secondId);
            Assert.Equal(2, store.ListDocuments(collectionId).Count);
            using var changedBytes = new MemoryStream(Encoding.UTF8.GetBytes("orchard updated north"));
            var changed = await store.ImportAsync(collectionId, "notes.txt", changedBytes, "text/plain", first.RelativePath);
            Assert.Equal(firstId, changed.Id);
            Assert.Equal(2, changed.Version);
            Assert.Equal(1, store.ListDocuments(collectionId).Single(document => document.Id == secondId).Version);
            Assert.Equal("Project/north/notes.txt", Assert.Single(await Search(store, collectionId, "updated")).RelativePath);
        }
        using var reopened = new KnowledgeStore(_root, new TextExtractor());
        Assert.Equal(new[] { "Project/north/notes.txt", "Project/south/notes.txt" },
            reopened.ListDocuments(collectionId).Select(document => document.RelativePath));
        Assert.Equal("Project/south/notes.txt", (await reopened.ReindexAsync(collectionId, secondId)).RelativePath);
        using var rootBytes = new MemoryStream(Encoding.UTF8.GetBytes("orchard root"));
        var root = await reopened.ImportAsync(collectionId, "notes.txt", rootBytes, "text/plain");
        Assert.DoesNotContain(root.Id, new[] { firstId, secondId });
        Assert.Equal(3, reopened.ListDocuments(collectionId).Count);
    }

    [Theory]
    [InlineData("/Folder/notes.txt")]
    [InlineData("Folder/../notes.txt")]
    [InlineData("Folder/./notes.txt")]
    [InlineData("Folder//notes.txt")]
    [InlineData("C:/Folder/notes.txt")]
    [InlineData("Folder\\notes.txt")]
    [InlineData("Folder/other.txt")]
    [InlineData("Folder/notes.txt\n")]
    public async Task FolderPathsRejectTraversalAbsolutePathsAndMismatchedUploadNames(string path)
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        var collection = await store.CreateCollectionAsync("Project");
        using var bytes = new MemoryStream(Encoding.UTF8.GetBytes("private notes"));
        await Assert.ThrowsAsync<ArgumentException>(() => store.ImportAsync(collection.Id, "notes.txt", bytes, "text/plain", path));
        Assert.Empty(store.ListDocuments(collection.Id));
        Assert.Equal(0, bytes.Position);
    }

    [Fact]
    public async Task FolderPathsPreserveUnicodeSpacesAndColonsWithinFolderNames()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        var collection = await store.CreateCollectionAsync("Project");
        using var bytes = new MemoryStream(Encoding.UTF8.GetBytes("orchard plans"));
        var document = await store.ImportAsync(collection.Id, "notes.txt", bytes, "text/plain", "研究资料/session 10:30/notes.txt");
        Assert.Equal("研究资料/session 10:30/notes.txt", document.RelativePath);
        Assert.Equal(document.Id, Assert.Single(await Search(store, collection.Id, "orchard")).DocumentId);
    }

    [Fact]
    public async Task CollectionRenamePreservesScopeDescriptionsAndCitationsAcrossRestart()
    {
        KnowledgeCollection collection;
        string reference;
        using (var store = new KnowledgeStore(_root, new TextExtractor()))
        {
            collection = await store.CreateCollectionAsync("Before", "Shared plans");
            await Import(store, collection.Id, "notes.txt", "orchard blue");
            reference = Assert.Single(await Search(store, collection.Id, "orchard")).Reference;
            var renamed = await store.RenameCollectionAsync(collection.Id, "  After  ");
            Assert.Equal(collection with { Name = "After" }, renamed);
            Assert.Equal("orchard blue", Assert.Single(store.Read(reference)).Text);
            await Assert.ThrowsAsync<ArgumentException>(() => store.RenameCollectionAsync(collection.Id, " "));
            await Assert.ThrowsAsync<ArgumentException>(() => store.RenameCollectionAsync(collection.Id, new string('a', 201)));
            await Assert.ThrowsAsync<KeyNotFoundException>(() => store.RenameCollectionAsync(new string('a', 32), "Missing"));
        }
        using var reopened = new KnowledgeStore(_root, new TextExtractor());
        Assert.Equal(collection with { Name = "After" }, Assert.Single(reopened.ListCollections()));
        Assert.Equal(reference, Assert.Single(await Search(reopened, collection.Id, "orchard")).Reference);
    }

    [Fact]
    public async Task SourcePreviewIsBoundedVersionCheckedAndExcludesMetadataPlaceholder()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor(), options: new()
        { ChunkCharacters = 16, ChunkOverlapCharacters = 3 });
        var collection = await store.CreateCollectionAsync("Sources");
        var document = await Import(store, collection.Id, "notes.txt", "alpha beta gamma delta epsilon zeta eta theta");
        var preview = store.PreviewDocument(collection.Id, document.Id, 1, 24);
        Assert.StartsWith("alpha beta gamma", preview.Text);
        Assert.True(preview.Text.Length <= 24);
        Assert.True(preview.Truncated);
        Assert.Equal(document.Id, preview.Document.Id);
        var complete = store.PreviewDocument(collection.Id, document.Id, 1);
        Assert.Equal("alpha beta gamma delta epsilon zeta eta theta", complete.Text);
        Assert.False(complete.Truncated);
        await store.ReindexAsync(collection.Id, document.Id);
        Assert.Throws<KnowledgeStaleReferenceException>(() => store.PreviewDocument(collection.Id, document.Id, 1));
        Assert.Throws<KeyNotFoundException>(() => store.PreviewDocument(collection.Id, new string('a', 32), 1));

        using var metadataStore = new KnowledgeStore(Path.Combine(_root, "metadata"), new DelegateExtractor((_, _) =>
            Task.FromResult(new KnowledgeExtraction { Status = KnowledgeDocumentStatus.MetadataOnly })));
        var mediaCollection = await metadataStore.CreateCollectionAsync("Media");
        var media = await Import(metadataStore, mediaCollection.Id, "recording.wav", "fixture");
        Assert.Equal("", metadataStore.PreviewDocument(mediaCollection.Id, media.Id, 1).Text);
        Assert.Equal("media", media.Kind);
    }

    [Fact]
    public async Task SourcePreviewStartsAtCitedChunkAndNavigatesWithoutDroppingIndexedExcerpts()
    {
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(new KnowledgeExtraction
        {
            Segments = [new(new string('a', 23000), new() { Page = 1 }, "text"),
                new(new string('b', 23000), new() { Page = 2 }, "text"),
                new(new string('c', 23000), new() { Page = 3 }, "text")],
        })), options: new() { ChunkCharacters = 32000, ChunkOverlapCharacters = 0 });
        var collection = await store.CreateCollectionAsync("Pages");
        var document = await Import(store, collection.Id, "pages.pdf", "fixture");
        var first = store.PreviewDocument(collection.Id, document.Id, 1);
        Assert.Equal(new string('a', 23000), first.Text);
        Assert.Equal(1, first.NextChunkIndex);
        Assert.Null(first.PreviousChunkIndex);
        Assert.Equal(1, first.Location!.Page);
        var next = store.PreviewDocument(collection.Id, document.Id, 1, chunkIndex: first.NextChunkIndex!.Value);
        Assert.Equal(new string('b', 23000), next.Text);
        Assert.Equal(0, next.PreviousChunkIndex);
        Assert.Equal(2, next.NextChunkIndex);
        var cited = store.PreviewDocument(collection.Id, document.Id, 1, chunkIndex: 2);
        Assert.Equal(new string('c', 23000), cited.Text);
        Assert.Equal(3, cited.Location!.Page);
        Assert.Equal(1, cited.PreviousChunkIndex);
        Assert.Null(cited.NextChunkIndex);
        Assert.False(cited.Truncated);
        Assert.Throws<KeyNotFoundException>(() => store.PreviewDocument(collection.Id, document.Id, 1, chunkIndex: 3));
    }

    [Fact]
    public async Task DurableImportSearchReadAndOriginalSurviveRestartWithoutReextraction()
    {
        var extractor = new TextExtractor();
        KnowledgeCollection collection;
        KnowledgeDocument imported;
        string reference;
        using (var store = new KnowledgeStore(_root, extractor))
        {
            collection = await store.CreateCollectionAsync("Engineering", "Runbooks");
            imported = await Import(store, collection.Id, "runbook.md", "Quasar release approval requires violet telemetry.");
            var hit = Assert.Single(await Search(store, collection.Id, "quasar violet"));
            reference = hit.Reference;
            Assert.Equal(imported.Id, hit.DocumentId);
            Assert.Equal(KnowledgeDocumentStatus.Ready, hit.Status);
            Assert.Equal("text-v1", imported.ExtractorId);
            Assert.Contains("telemetry", Assert.Single(store.Read(reference, collectionIds: [collection.Id])).Text);
        }
        using (var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => throw new Exception("Restart must not extract"))))
        {
            Assert.Equal(collection, Assert.Single(store.ListCollections()));
            Assert.Equal(imported.ContentHash, Assert.Single(store.ListDocuments(collection.Id)).ContentHash);
            Assert.Equal(reference, Assert.Single(await Search(store, collection.Id, "violet")).Reference);
            using var original = new StreamReader(store.OpenOriginal(collection.Id, imported.Id, imported.Version));
            Assert.Equal("Quasar release approval requires violet telemetry.", await original.ReadToEndAsync());
        }
        Assert.Equal(1, extractor.Calls);
    }

    [Fact]
    public async Task IdempotentImportUpdateReindexAndDeletionInvalidateOldReferences()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        var collection = await store.CreateCollectionAsync("Notes");
        var first = await Import(store, collection.Id, "a.txt", "Old zephyr protocol.");
        string reference = Assert.Single(await Search(store, collection.Id, "zephyr")).Reference;
        var duplicate = await Import(store, collection.Id, "a.txt", "Old zephyr protocol.");
        Assert.Equal(first.Id, duplicate.Id);
        Assert.Equal(1, duplicate.Version);
        var updated = await Import(store, collection.Id, "a.txt", "Replacement quartz protocol.");
        Assert.Equal(first.Id, updated.Id);
        Assert.Equal(2, updated.Version);
        Assert.Empty(await Search(store, collection.Id, "zephyr"));
        Assert.Throws<KnowledgeStaleReferenceException>(() => store.Read(reference));
        Assert.Throws<KnowledgeStaleReferenceException>(() => store.OpenOriginal(collection.Id, first.Id, 1));
        var current = Assert.Single(await Search(store, collection.Id, "quartz"));
        var reindexed = await store.ReindexAsync(collection.Id, first.Id);
        Assert.Equal(3, reindexed.Version);
        Assert.Equal(updated.ContentHash, reindexed.ContentHash);
        Assert.Throws<KnowledgeStaleReferenceException>(() => store.Read(current.Reference));
        Assert.True(await store.DeleteDocumentAsync(collection.Id, first.Id));
        Assert.False(await store.DeleteDocumentAsync(collection.Id, first.Id));
        Assert.Empty(store.ListDocuments(collection.Id));
        Assert.Empty(await Search(store, collection.Id, "quartz"));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "objects")));
    }

    [Fact]
    public async Task CollectionScopeFiltersBeforeRankingAndReferenceResolution()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        var privateCollection = await store.CreateCollectionAsync("Private");
        var selected = await store.CreateCollectionAsync("Selected");
        await Import(store, privateCollection.Id, "private.txt", "supersecret supersecret code 9274");
        await Import(store, selected.Id, "public.txt", "supersecret is a placeholder in this document");
        var privateHit = Assert.Single(await Search(store, privateCollection.Id, "9274"));
        var selectedHit = Assert.Single(await Search(store, selected.Id, "supersecret"));
        Assert.Equal("public.txt", selectedHit.FileName);
        Assert.Empty(await Search(store, selected.Id, "9274"));
        Assert.Empty(await store.SearchAsync("supersecret", new()));
        Assert.Throws<KeyNotFoundException>(() => store.Read(privateHit.Reference, collectionIds: [selected.Id]));
        Assert.Throws<KeyNotFoundException>(() => store.Read(privateHit.Reference, collectionIds: []));
        Assert.Throws<KeyNotFoundException>(() => store.OpenOriginal(selected.Id, privateHit.DocumentId, 1));
        Assert.False(await store.DeleteDocumentAsync(selected.Id, privateHit.DocumentId));
    }

    [Fact]
    public async Task Bm25RanksFocusedDocumentAboveLongNoisyMatchAndHandlesCjk()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor(), options: new() { ChunkCharacters = 10000 });
        var collection = await store.CreateCollectionAsync("Manuals");
        await Import(store, collection.Id, "focused.txt", "helium cooling specification");
        await Import(store, collection.Id, "noise.txt", "helium " + string.Join(" ", Enumerable.Repeat("irrelevant", 500)));
        await Import(store, collection.Id, "chinese.txt", "知识系统支持中文资料检索。");
        var results = await Search(store, collection.Id, "helium cooling");
        Assert.Equal("focused.txt", results[0].FileName);
        Assert.Equal("chinese.txt", Assert.Single(await Search(store, collection.Id, "知识")).FileName);
    }

    [Fact]
    public async Task ChineseQuestionFindsEnglishNamedSourceInsteadOfUnrelatedCharacterMatchesAfterRestart()
    {
        string collectionId;
        using (var store = new KnowledgeStore(_root, new TextExtractor()))
        {
            collectionId = (await store.CreateCollectionAsync("School documents")).Id;
            await Import(store, collectionId, "report.txt", "Jonathan: Mathematics 4; Writing 2. Teacher comments: strong reasoning; needs clearer structure.");
            for (int i = 0; i < 12; i++)
                await Import(store, collectionId, $"unrelated-{i}.txt", "这是一个关于系统的介绍。其中包括较多的功能和综合资料。出点不足以成为老师的评价标准。");
        }
        using var reopened = new KnowledgeStore(_root, new TextExtractor());
        var results = await Search(reopened, collectionId,
            "Jonathan的哪个成绩比较好，哪个成绩比较差。请综合老师给他的评语，找出其中的亮点和不足。");
        Assert.Equal("report.txt", results[0].FileName);
        Assert.Contains("Mathematics 4", results[0].Text);
    }

    [Fact]
    public async Task MixedScriptQuestionPrioritizesNamedPersonOverOtherStudentsMatchingChineseTopics()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        var collection = await store.CreateCollectionAsync("School");
        await Import(store, collection.Id, "jonathan.txt", "Jonathan shows strong mathematical reasoning and needs to improve writing.");
        await Import(store, collection.Id, "other-student.txt", "另一位同学的成绩评语：数学成绩很好，老师评语指出写作成绩需要提高。");
        Assert.Equal("jonathan.txt", (await Search(store, collection.Id, "Jonathan 成绩 评语"))[0].FileName);
    }

    [Theory]
    [InlineData("知识", "知识系统资料。", "知其所以然，学而后识。")]
    [InlineData("𠀀文", "𠀀文资料。", "𠀀，其他文字。")]
    public async Task CjkPhrasesRequireAdjacentCharactersWhileSingleCharacterQueriesRemainSearchable(string query, string relevant, string noise)
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        var collection = await store.CreateCollectionAsync("CJK");
        await Import(store, collection.Id, "target.txt", relevant);
        await Import(store, collection.Id, "noise.txt", noise);
        Assert.Equal("target.txt", Assert.Single(await Search(store, collection.Id, query)).FileName);
        string firstRune = query.EnumerateRunes().First().ToString();
        Assert.Equal(2, (await Search(store, collection.Id, firstRune)).Count);
    }

    [Fact]
    public async Task SearchSuppressesDuplicatePageRepresentationsButKeepsChangedScoresAndOtherPages()
    {
        const string paragraph = "Jonathan demonstrates clear mathematical reasoning and consistently explains the steps used to solve challenging problems. The teacher recommends continued practice with written explanations and independent revision. During group projects he listens carefully to the ideas shared by other students and contributes thoughtful suggestions for improving the final presentation. He checks his work and supports classmates as they develop their own understanding. Score 4.";
        var extraction = new KnowledgeExtraction { Segments = [
            new(paragraph, new() { Page = 1 }, "text"),
            new(paragraph.Replace("teacher recommends", "teacher suggests"), new() { Page = 1 }, "ocr"),
            new(paragraph.Replace("Score 4", "Score 2"), new() { Page = 1 }, "ocr"),
            new(paragraph, new() { Page = 2 }, "text"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)));
        var collection = await store.CreateCollectionAsync("Duplicates");
        await Import(store, collection.Id, "report.pdf", "fixture");
        var results = await Search(store, collection.Id, "Jonathan mathematical reasoning");
        Assert.Equal(3, results.Count);
        Assert.Contains(results, result => result.Location?.Page == 2);
        Assert.Contains(results, result => result.Text.Contains("Score 2"));
        Assert.All(results, result => Assert.Equal(4, result.DocumentChunkCount));
        // Suppressed search duplicates remain explicitly readable by their stable
        // reference if an OCR difference or original page needs inspection.
        string reference = results[0].Reference[..(results[0].Reference.LastIndexOf('/') + 1)] + "1";
        Assert.Contains("teacher suggests", Assert.Single(store.Read(reference)).Text);
    }

    [Theory]
    [InlineData("Jonathan的成绩和老师评语")]
    [InlineData("Jonathan strengths weaknesses")]
    public async Task BroadOverviewDiversifiesRelevantPagesWithoutTreatingPageNumberAsRelevance(string query)
    {
        var extraction = new KnowledgeExtraction { Segments = [
            new("Jonathan. Report card scale: 3 meets standard, 4 exceeds standard. Mathematics 4; writing 3.", new() { Page = 1 }, "text"),
            new("Jonathan. Jonathan. Jonathan. A repeated footer unrelated to grades.", new() { Page = 1 }, "text"),
            new("Jonathan. Computer science 4, keyboarding 3.5. Spanish reading 4.", new() { Page = 2 }, "text"),
            new("Jonathan participates well. Jonathan helps others. Jonathan shows enthusiasm. Jonathan enjoys school.", new() { Page = 3 }, "text"),
            new("Jonathan participates well. Jonathan helps others. Jonathan shows enthusiasm. Jonathan enjoys school.", new() { Page = 3 }, "ocr"),
            new("Jonathan has made progress. Jonathan is an engaged learner. Jonathan works well. Jonathan is curious.", new() { Page = 3 }, "text"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((context, _) => Task.FromResult(context.FileName == "other.txt"
            ? new KnowledgeExtraction { Segments = [new("另一位同学成绩评语：老师的评语和成绩显示进步。")] } : extraction)));
        var collection = await store.CreateCollectionAsync("Report");
        await Import(store, collection.Id, "report.pdf", "fixture");
        await Import(store, collection.Id, "other.txt", "other fixture");
        var results = await store.SearchAsync(query, new() { CollectionIds = [collection.Id], Limit = 3 });
        Assert.Equal(new int?[] { 1, 2, 3 }, results.Select(result => result.Location?.Page).Order());
        Assert.Equal(results.Select(result => result.Score).OrderDescending(), results.Select(result => result.Score));
    }

    [Fact]
    public async Task BroadCoverageDoesNotPromoteDenseOnlyPagesAheadOfLexicalEvidence()
    {
        var extraction = new KnowledgeExtraction { Segments = [
            new("Jonathan mathematics results", new() { Page = 1 }),
            new("Unrelated school lunch information", new() { Page = 2 }),
            new("Jonathan teacher comments", new() { Page = 3 }),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)), new VectorProvider());
        var collection = await store.CreateCollectionAsync("Hybrid coverage");
        await Import(store, collection.Id, "report.pdf", "fixture");
        var results = await store.SearchAsync("Jonathan strengths weaknesses", new() { CollectionIds = [collection.Id], Limit = 2 });
        Assert.Equal(new int?[] { 1, 3 }, results.Select(result => result.Location?.Page));
    }

    [Theory]
    [InlineData("contraindications")]
    [InlineData("contraindications for children")]
    public async Task FocusedQueryPrefersDetailedLaterPageEvenWhenOtherQueryTermsAreAbsent(string query)
    {
        var extraction = new KnowledgeExtraction { Segments = [
            new("Introduction: consult the contraindications section later in this handbook. "
                + string.Join(' ', Enumerable.Repeat("general background information", 60)), new() { Page = 1 }),
            new("Contraindications: the contraindications checklist and contraindications assessment are explained here.", new() { Page = 8 }),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)));
        var collection = await store.CreateCollectionAsync("Focused lookup");
        await Import(store, collection.Id, "handbook.pdf", "fixture");
        var results = await store.SearchAsync(query, new() { CollectionIds = [collection.Id], Limit = 1 });
        Assert.Equal(8, Assert.Single(results).Location?.Page);
    }

    [Theory]
    [InlineData("Orion 故障诊断")]
    [InlineData("Orion diagnostic overview")]
    public async Task BroadMixedLanguageQueryKeepsStrongerLaterEvidenceFirst(string query)
    {
        var extraction = new KnowledgeExtraction { Segments = [
            new("Orion is briefly mentioned in this general introduction. "
                + string.Join(' ', Enumerable.Repeat("background and unrelated system details", 30)), new() { Page = 1 }, "text"),
            new("Orion troubleshooting: Orion fault codes and Orion recovery steps are explained here.", new() { Page = 9 }, "text"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)));
        var collection = await store.CreateCollectionAsync("System handbook");
        await Import(store, collection.Id, "system.pdf", "fixture");
        var results = await store.SearchAsync(query, new() { CollectionIds = [collection.Id], Limit = 2 });
        Assert.Equal(new int?[] { 9, 1 }, results.Select(result => result.Location?.Page));
        Assert.True(results[0].Score >= results[1].Score);
    }

    [Fact]
    public async Task FocusedMultiTermRelevanceCanReturnTwoStrongExcerptsFromOnePage()
    {
        var extraction = new KnowledgeExtraction { Segments = [
            new("Alarm reset: press the reset switch when the alarm flashes.", new() { Page = 4 }, "text"),
            new("Alarm reset troubleshooting: a reset clears the alarm after repair.", new() { Page = 4 }, "text"),
            new("Alarm and reset appear in this index. " + string.Join(' ', Enumerable.Repeat("general reference material", 40)), new() { Page = 1 }, "text"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)));
        var collection = await store.CreateCollectionAsync("Service manual");
        await Import(store, collection.Id, "service.pdf", "fixture");
        var results = await store.SearchAsync("alarm reset", new() { CollectionIds = [collection.Id], Limit = 2 });
        Assert.Equal(new int?[] { 4, 4 }, results.Select(result => result.Location?.Page));
        Assert.True(results[0].Score >= results[1].Score);
    }

    [Fact]
    public async Task ChunkingRetainsLocationsAndCharacterBudgetsWithoutBreakingSurrogates()
    {
        var extraction = new KnowledgeExtraction
        {
            Segments = [new(new string('A', 50) + " 😀 " + new string('B', 60), new() { Page = 7 }, "text"),
                new("At fifty seconds the alarm sounded.", new() { StartSeconds = 50, EndSeconds = 53 }, "audio")],
            Status = KnowledgeDocumentStatus.Partial, Warnings = ["One unreadable page"]
        };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)),
            options: new() { ChunkCharacters = 64, ChunkOverlapCharacters = 8 });
        var collection = await store.CreateCollectionAsync("Evidence");
        var doc = await Import(store, collection.Id, "recording.bin", "anything");
        Assert.True(doc.ChunkCount >= 3);
        var hits = await Search(store, collection.Id, "alarm");
        var audio = Assert.Single(hits);
        Assert.Equal("audio", audio.Modality);
        Assert.Equal(50, audio.Location!.StartSeconds);
        Assert.Equal(KnowledgeDocumentStatus.Partial, audio.Status);
        var bounded = store.Read(audio.Reference, 11, adjacentChunks: 2, collectionIds: [collection.Id]);
        Assert.True(bounded.Sum(x => x.Text.Length) <= 11);
        Assert.True(bounded.Last().Truncated);
        Assert.All(bounded, r => Assert.False(r.Text.Length > 0 && char.IsHighSurrogate(r.Text[^1])));
    }

    [Fact]
    public async Task MetadataOnlyAssetsAreExplicitAndNeverEmbeddedAsSemanticContent()
    {
        var provider = new VectorProvider();
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(new KnowledgeExtraction
        { Status = KnowledgeDocumentStatus.MetadataOnly, Warnings = ["No image description provider configured."] })), provider);
        var collection = await store.CreateCollectionAsync("Photos");
        var doc = await Import(store, collection.Id, "nebula.png", "fake image bytes");
        Assert.Equal(KnowledgeDocumentStatus.MetadataOnly, doc.Status);
        Assert.Null(doc.EmbeddingModelId);
        var hit = Assert.Single(await Search(store, collection.Id, "nebula"));
        Assert.Equal("metadata", hit.Modality);
        Assert.Equal(0, provider.Calls);
    }

    [Fact]
    public async Task ExtractedAssetsKeepNamesAndFolderPathsSearchableWithoutChangingEvidenceAcrossRestart()
    {
        const string caption = "An abstract blue illustration with scattered glowing cubes.";
        var extraction = new KnowledgeExtraction { Segments = [
            new(caption, new() { Label = "Image description" }, "image"),
            new("Small lettering at the bottom.", new() { Label = "Visible text" }, "ocr"),
        ] };
        string collectionId, documentId, reference;
        using (var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction))))
        {
            collectionId = (await store.CreateCollectionAsync("Media")).Id;
            using var bytes = new MemoryStream(Encoding.UTF8.GetBytes("image fixture"));
            var document = await store.ImportAsync(collectionId, "banner_1.png", bytes, "image/png", "TensorSharp/promotions/banner_1.png");
            documentId = document.Id;
            var byName = Assert.Single(await Search(store, collectionId, "banner_1.png"));
            Assert.Equal(caption, byName.Text);
            Assert.Equal("image", byName.Modality);
            Assert.Equal(0, byName.ChunkIndex);
            Assert.Equal(2, byName.DocumentChunkCount);
            Assert.Equal(documentId, Assert.Single(await Search(store, collectionId, "请到知识库中寻找tensorsharp的宣传画")).DocumentId);
            await store.ReindexAsync(collectionId, documentId);
            reference = Assert.Single(await Search(store, collectionId, "banner")).Reference;
        }
        using var reopened = new KnowledgeStore(_root, new DelegateExtractor((_, _) => throw new Exception("Restart must not re-extract")));
        Assert.Equal(reference, Assert.Single(await Search(reopened, collectionId, "banner")).Reference);
        var byFolder = Assert.Single(await Search(reopened, collectionId, "TensorSharp promotions"));
        Assert.Equal(documentId, byFolder.DocumentId);
        Assert.Equal(caption, byFolder.Text);
        Assert.Equal(caption, Assert.Single(reopened.Read(byFolder.Reference)).Text);
        Assert.DoesNotContain("banner", reopened.PreviewDocument(collectionId, documentId, 2).Text);
        await reopened.DeleteDocumentAsync(collectionId, documentId);
        Assert.Empty(await Search(reopened, collectionId, "TensorSharp banner"));
    }

    [Theory]
    [InlineData("请到知识库中寻找tensorsharp的宣传画")]
    [InlineData("Please find the TensorSharp poster")]
    public async Task ExplicitImageLookupSurfacesMatchingOriginalBeforeRepeatedCodeChunks(string query)
    {
        var caption = "A woman on a white background with blue geometric decorations. Visible text: TensorSharp. Open-Source Deep Learning/AI C# Library.";
        var extractor = new DelegateExtractor((context, _) => Task.FromResult(context.FileName switch
        {
            "banner_1.png" => new KnowledgeExtraction { Segments = [new(caption, new() { Label = "Image description and visible text" }, "image")], Status = KnowledgeDocumentStatus.Partial },
            "unrelated.png" => new KnowledgeExtraction { Segments = [new("An unrelated promotional poster. 宣传画。", Modality: "image")] },
            "manual.pdf" => new KnowledgeExtraction { Segments = [new("TensorSharp image poster API examples", new() { Page = 1 }, "ocr")] },
            "source.txt" => new KnowledgeExtraction { Segments = Enumerable.Range(0, 100).Select(index => new KnowledgeSegment($"TensorSharp namespace Code{index}", Modality: "text")).ToArray() },
            _ => new KnowledgeExtraction { Segments = [new("TensorSharp project readme.", Modality: "text")] },
        }));
        using var store = new KnowledgeStore(_root, extractor);
        var collection = await store.CreateCollectionAsync("test");
        var otherCollection = await store.CreateCollectionAsync("unselected");
        await Import(store, collection.Id, "source.txt", "code fixture");
        await Import(store, collection.Id, "readme.txt", "readme fixture");
        await Import(store, collection.Id, "manual.pdf", "pdf fixture");
        await Import(store, collection.Id, "unrelated.png", "unrelated image fixture");
        var image = await Import(store, collection.Id, "banner_1.png", "image fixture");
        var hiddenImage = await Import(store, otherCollection.Id, "banner_1.png", "hidden fixture");

        var results = await store.SearchAsync(query, new() { CollectionIds = [collection.Id], Limit = 4 });
        Assert.Equal(image.Id, results[0].DocumentId);
        Assert.Equal("image", results[0].Modality);
        Assert.Equal(caption, results[0].Text);
        Assert.DoesNotContain(results, result => result.DocumentId == hiddenImage.Id);
        Assert.Contains(results, result => result.FileName == "readme.txt");
        Assert.Contains(results, result => result.FileName == "manual.pdf");
    }

    [Theory]
    [InlineData("请寻找TensorSharp的录音", "recording.wav", "audio")]
    [InlineData("Find the TensorSharp audio", "recording.wav", "audio")]
    [InlineData("请寻找TensorSharp的视频", "trailer.mp4", "video")]
    [InlineData("Find the TensorSharp video", "trailer.mp4", "video")]
    public async Task ExplicitMediaLookupUsesOriginalTypeInsteadOfExtractedCaptionModality(string query, string fileName, string requested)
    {
        using var store = new KnowledgeStore(_root, new DelegateExtractor((context, _) => Task.FromResult(new KnowledgeExtraction
        { Segments = [new(context.FileName == fileName ? "TensorSharp presentation." : "TensorSharp", Modality: context.FileName == fileName ? "image" : "text")] })));
        var collection = await store.CreateCollectionAsync("Media");
        await Import(store, collection.Id, "source.txt", "source fixture");
        using var bytes = new MemoryStream(Encoding.UTF8.GetBytes("media fixture"));
        var document = await store.ImportAsync(collection.Id, fileName, bytes, requested + "/fixture");
        Assert.Equal(document.Id, (await Search(store, collection.Id, query))[0].DocumentId);
    }

    [Theory]
    [InlineData("How does TensorSharp image encoding work?")]
    [InlineData("What model does TensorSharp use for image encoding?")]
    [InlineData("Find the TensorSharp image API documentation")]
    [InlineData("TensorSharp如何编码图片")]
    public async Task QuestionsAboutMediaProcessingKeepDocumentationRelevance(string query)
    {
        using var store = new KnowledgeStore(_root, new DelegateExtractor((context, _) => Task.FromResult(new KnowledgeExtraction
        {
            Segments = [new(context.FileName == "guide.pdf"
                ? "TensorSharp image encoding uses an encoder model. API documentation explains the work."
                : "TensorSharp image poster. " + string.Join(' ', Enumerable.Repeat("blue decorative background artwork", 40)),
                Modality: context.FileName == "guide.pdf" ? "ocr" : "image")],
        })));
        var collection = await store.CreateCollectionAsync("Media documentation");
        await Import(store, collection.Id, "banner_1.png", "image fixture");
        var guide = await Import(store, collection.Id, "guide.pdf", "guide fixture");
        Assert.Equal(guide.Id, (await Search(store, collection.Id, query))[0].DocumentId);
    }

    [Fact]
    public async Task DenseRetrievalFindsSemanticMatchAndNeverMixesEmbeddingModelSpaces()
    {
        string collectionId, documentId;
        using (var store = new KnowledgeStore(_root, new TextExtractor(), new VectorProvider("space-a")))
        {
            var collection = await store.CreateCollectionAsync("Things"); collectionId = collection.Id;
            documentId = (await Import(store, collection.Id, "a.txt", "automobile maintenance")).Id;
            await Import(store, collection.Id, "b.txt", "apple orchard");
            Assert.Equal("a.txt", Assert.Single(await Search(store, collection.Id, "car")).FileName);
        }
        var otherSpace = new VectorProvider("space-b");
        using (var store = new KnowledgeStore(_root, new TextExtractor(), otherSpace))
        {
            Assert.Empty(await Search(store, collectionId, "car"));
            Assert.Equal(0, otherSpace.Calls);
            Assert.Equal("a.txt", Assert.Single(await Search(store, collectionId, "automobile")).FileName);
            var doc = await store.ReindexAsync(collectionId, documentId);
            Assert.Equal("space-b", doc.EmbeddingModelId);
            Assert.Equal("a.txt", Assert.Single(await Search(store, collectionId, "car")).FileName);
        }
        Assert.True(otherSpace.Disposed);
    }

    [Theory]
    [InlineData("count")]
    [InlineData("dimension")]
    [InlineData("nan")]
    [InlineData("zero")]
    public async Task InvalidEmbeddingResponsesNeverPublishDocument(string invalid)
    {
        var provider = new InvalidVectorProvider(invalid);
        using var store = new KnowledgeStore(_root, new TextExtractor(), provider);
        var collection = await store.CreateCollectionAsync("Broken");
        await Assert.ThrowsAsync<InvalidDataException>(() => Import(store, collection.Id, "file.txt", "hello"));
        Assert.Empty(store.ListDocuments(collection.Id));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "objects")));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "staging")));
    }

    [Fact]
    public async Task FailedAndCancelledUpdatesKeepPreviouslyCommittedVersion()
    {
        var extractor = new TextExtractor();
        using var store = new KnowledgeStore(_root, extractor);
        var collection = await store.CreateCollectionAsync("Stable");
        var first = await Import(store, collection.Id, "a.txt", "committed jade record");
        extractor.Fail = true;
        await Assert.ThrowsAsync<InvalidDataException>(() => Import(store, collection.Id, "a.txt", "replacement"));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Import(store, collection.Id, "a.txt", "cancelled", cancelled.Token));
        Assert.Equal(first.Version, Assert.Single(store.ListDocuments(collection.Id)).Version);
        Assert.Equal(first.Id, Assert.Single(await Search(store, collection.Id, "jade")).DocumentId);
        Assert.Single(Directory.GetFiles(Path.Combine(_root, "objects")));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "staging")));
    }

    [Fact]
    public async Task CancellationDuringExtractionLeavesNoVisibleOrStagedDocument()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var store = new KnowledgeStore(_root, new DelegateExtractor(async (_, ct) =>
        { entered.SetResult(); await Task.Delay(Timeout.Infinite, ct); return new(); }));
        var collection = await store.CreateCollectionAsync("Cancelled");
        using var cancellation = new CancellationTokenSource();
        var pending = Import(store, collection.Id, "a.txt", "transient", cancellation.Token);
        await entered.Task; cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);
        Assert.Empty(store.ListDocuments(collection.Id));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "staging")));
    }

    [Fact]
    public async Task LimitsRejectOversizedStreamsAndExtractionWithoutSilentTruncation()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor(), options: new()
        { MaxFileBytes = 40, MaxExtractedCharacters = 20, ChunkCharacters = 16, ChunkOverlapCharacters = 0 });
        var collection = await store.CreateCollectionAsync("Bounded");
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => Import(store, collection.Id, "big.txt", new string('x', 41)));
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => Import(store, collection.Id, "text.txt", new string('x', 30)));
        Assert.Empty(store.ListDocuments(collection.Id));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "staging")));
    }

    [Fact]
    public async Task DocumentChunkAndStorageQuotasAllowReplacementButRejectGrowth()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor(), options: new()
        { MaxDocuments = 1, MaxStoredBytes = 20, MaxTotalChunks = 1, MaxChunksPerDocument = 1, ChunkCharacters = 16, ChunkOverlapCharacters = 0 });
        var collection = await store.CreateCollectionAsync("Bounded");
        await Import(store, collection.Id, "a.txt", "short");
        await Import(store, collection.Id, "a.txt", "changed");
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => Import(store, collection.Id, "b.txt", "other"));
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => Import(store, collection.Id, "a.txt", new string('x', 18)));
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => Import(store, collection.Id, "a.txt", new string('x', 21)));
        Assert.Equal(2, Assert.Single(store.ListDocuments(collection.Id)).Version);
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("/etc/passwd")]
    [InlineData("ABCDEF0123456789ABCDEF0123456789")]
    [InlineData("")]
    public async Task MalformedIdsCannotReachStorage(string id)
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        Assert.Throws<ArgumentException>(() => store.ListDocuments(id));
        await Assert.ThrowsAsync<ArgumentException>(() => Import(store, id, "a.txt", "data"));
        await Assert.ThrowsAsync<ArgumentException>(() => store.DeleteCollectionAsync(id));
    }

    [Theory]
    [InlineData("../secret.txt")]
    [InlineData("folder\\secret.txt")]
    [InlineData("name\n.txt")]
    public async Task DisplayNamesCannotCarryPathsOrControlCharacters(string name)
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        var collection = await store.CreateCollectionAsync("Files");
        await Assert.ThrowsAsync<ArgumentException>(() => Import(store, collection.Id, name, "data"));
    }

    [Theory]
    [InlineData("file:///etc/passwd")]
    [InlineData("kb://../x@1/0")]
    [InlineData("kb://00000000000000000000000000000000/00000000000000000000000000000000@1/-1")]
    [InlineData("kb://00000000000000000000000000000000/00000000000000000000000000000000@1/0?x")]
    public void InvalidSourceReferencesAreRejected(string reference)
    {
        using var store = new KnowledgeStore(_root, new TextExtractor());
        Assert.Throws<ArgumentException>(() => store.Read(reference));
    }

    [Fact]
    public async Task ConcurrentImportsSerializeAndSingleWriterLockSurvivesFailedSecondOpen()
    {
        using (var store = new KnowledgeStore(_root, new TextExtractor()))
        {
            Assert.Throws<IOException>(() => new KnowledgeStore(_root, new TextExtractor()));
            var collection = await store.CreateCollectionAsync("Concurrent");
            var docs = await Task.WhenAll(Enumerable.Range(0, 30).Select(i => Import(store, collection.Id, $"{i}.txt", $"unique{i} marker")));
            Assert.Equal(30, store.ListDocuments(collection.Id).Count);
            Assert.Equal(30, docs.Select(d => d.Id).Distinct().Count());
            Assert.Equal("17.txt", Assert.Single(await Search(store, collection.Id, "unique17")).FileName);
        }
        using var reopened = new KnowledgeStore(_root, new TextExtractor());
        Assert.Equal(30, reopened.ListDocuments(Assert.Single(reopened.ListCollections()).Id).Count);
    }

    [Fact]
    public async Task DeletingCollectionPreservesOtherCollectionAndSharedOriginal()
    {
        using (var store = new KnowledgeStore(_root, new TextExtractor()))
        {
            var first = await store.CreateCollectionAsync("First");
            var second = await store.CreateCollectionAsync("Second");
            await Import(store, first.Id, "first.txt", "shared durable text");
            var survivor = await Import(store, second.Id, "second.txt", "shared durable text");
            Assert.Single(Directory.GetFiles(Path.Combine(_root, "objects")));
            Assert.True(await store.DeleteCollectionAsync(first.Id));
            Assert.False(await store.DeleteCollectionAsync(first.Id));
            Assert.Single(Directory.GetFiles(Path.Combine(_root, "objects")));
            using var original = store.OpenOriginal(second.Id, survivor.Id, 1);
            Assert.Equal(survivor.SizeBytes, original.Length);
        }
        using var reopened = new KnowledgeStore(_root, new TextExtractor());
        Assert.Equal("Second", Assert.Single(reopened.ListCollections()).Name);
    }

    [Fact]
    public async Task StartupRecoversInterruptedCollectionDeletionAndOrphanBlobs()
    {
        string collectionId;
        using (var store = new KnowledgeStore(_root, new TextExtractor()))
        {
            collectionId = (await store.CreateCollectionAsync("Interrupted")).Id;
            await Import(store, collectionId, "a.txt", "orphan after deletion");
        }
        File.Delete(Path.Combine(_root, "collections", collectionId + ".json"));
        File.WriteAllText(Path.Combine(_root, "staging", "abandoned.upload"), "interrupted");
        using var reopened = new KnowledgeStore(_root, new TextExtractor());
        Assert.Empty(reopened.ListCollections());
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "documents")));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "objects")));
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "staging")));
    }

    [Fact]
    public async Task CorruptManifestFailsExplicitlyAndReleasesWriterLock()
    {
        using (var store = new KnowledgeStore(_root, new TextExtractor()))
        {
            var collection = await store.CreateCollectionAsync("Corrupt");
            await Import(store, collection.Id, "a.txt", "stored");
        }
        string manifest = Assert.Single(Directory.GetFiles(Path.Combine(_root, "documents")));
        string valid = File.ReadAllText(manifest);
        JsonNode node = JsonNode.Parse(valid)!;
        node["Schema"] = 999;
        File.WriteAllText(manifest, node.ToJsonString());
        Assert.Throws<InvalidDataException>(() => new KnowledgeStore(_root, new TextExtractor()));
        File.WriteAllText(manifest, valid);
        using var repaired = new KnowledgeStore(_root, new TextExtractor());
        Assert.Single(repaired.ListCollections());
    }

    [Fact]
    public async Task ProviderIsNotDisposedUntilInFlightSearchCompletes()
    {
        var provider = new BlockingQueryProvider();
        var store = new KnowledgeStore(_root, new TextExtractor(), provider);
        var collection = await store.CreateCollectionAsync("Lifecycle");
        await Import(store, collection.Id, "car.txt", "automobile maintenance");
        var search = Search(store, collection.Id, "car");
        await provider.Entered.Task;
        var disposing = Task.Run(store.Dispose);
        await Task.Delay(30);
        Assert.False(provider.Disposed);
        provider.Release.SetResult();
        Assert.Single(await search);
        await disposing;
        Assert.True(provider.Disposed);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => Search(store, collection.Id, "car"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task QueryEmbeddingAwaitCannotDiscloseDeletedOrReplacedSnapshot(bool replace)
    {
        var provider = new BlockingQueryProvider();
        using var store = new KnowledgeStore(_root, new TextExtractor(), provider);
        var collection = await store.CreateCollectionAsync("Revocation");
        var document = await Import(store, collection.Id, "car.txt", "automobile old confidential content");
        var pending = Search(store, collection.Id, "car");
        await provider.Entered.Task;
        if (replace) await Import(store, collection.Id, "car.txt", "apple replacement public content");
        else await store.DeleteDocumentAsync(collection.Id, document.Id);
        provider.Release.SetResult();
        Assert.Empty(await pending);
    }

    [Fact]
    public async Task ReadBudgetAlwaysIncludesRequestedChunkBeforeAdjacentContext()
    {
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(new KnowledgeExtraction
            { Segments = [new("earlier irrelevant text"), new("requested important text"), new("later extra text")] })));
        var collection = await store.CreateCollectionAsync("Read");
        await Import(store, collection.Id, "a.txt", "fixture");
        var result = Assert.Single(await Search(store, collection.Id, "requested"));
        var read = Assert.Single(store.Read(result.Reference, maxCharacters: 5, adjacentChunks: 2, collectionIds: [collection.Id]));
        Assert.Equal(result.ChunkIndex, read.ChunkIndex);
        Assert.Equal("reque", read.Text);
        var context = store.Read(result.Reference, adjacentChunks: 2, collectionIds: [collection.Id]);
        Assert.Equal(result.ChunkIndex, context[0].ChunkIndex);
        Assert.Equal(3, context.Count);
        Assert.All(context, chunk => Assert.Equal(3, chunk.DocumentChunkCount));
    }

    [Fact]
    public async Task DocumentContextIncludesUnmatchedLaterEvidenceWithExactProvenanceAndNoPartialReturn()
    {
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(new KnowledgeExtraction
        { Segments = [new("Aster overview: performance 4.", new() { Page = 1 }, "text"),
            new("The reviewer recommends clearer written explanations and checking each result.", new() { Page = 2 }, "text")] })));
        var collection = await store.CreateCollectionAsync("Whole document");
        await Import(store, collection.Id, "review.pdf", "fixture");
        var anchor = Assert.Single(await Search(store, collection.Id, "Aster"));
        var context = Assert.IsAssignableFrom<IReadOnlyList<KnowledgeSearchResult>>(
            store.ReadDocumentContext(anchor.Reference, collectionIds: [collection.Id]));
        Assert.Equal(new int?[] { 1, 2 }, context.Select(chunk => chunk.Location?.Page));
        Assert.Contains("clearer written explanations", context[1].Text);
        Assert.All(context, chunk =>
        {
            Assert.False(chunk.Truncated);
            Assert.Equal(Assert.Single(store.Read(chunk.Reference)).Text, chunk.Text);
            Assert.Equal(anchor.DocumentId, chunk.DocumentId);
            Assert.Equal(2, chunk.DocumentChunkCount);
        });
        int exactBudget = context.Sum(chunk => chunk.Text.Length);
        Assert.NotNull(store.ReadDocumentContext(anchor.Reference, exactBudget));
        Assert.Null(store.ReadDocumentContext(anchor.Reference, exactBudget - 1));
    }

    [Fact]
    public async Task DocumentContextDeduplicatesEquivalentPageTextAcrossDifferentChunkBoundaries()
    {
        string paragraph = string.Join(' ', Enumerable.Range(0, 40).Select(i => $"Observation{i}: clear reasoning."));
        var extraction = new KnowledgeExtraction { Segments = [
            new(paragraph, new() { Page = 1 }, "text"),
            new(paragraph.Replace(" ", "  \n"), new() { Page = 1 }, "ocr"),
            new(paragraph, new() { Page = 2 }, "text"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)),
            options: new() { ChunkCharacters = 150, ChunkOverlapCharacters = 20 });
        var collection = await store.CreateCollectionAsync("Representations");
        var document = await Import(store, collection.Id, "review.pdf", "fixture");
        string reference = $"kb://{collection.Id}/{document.Id}@{document.Version}/0";
        var context = Assert.IsAssignableFrom<IReadOnlyList<KnowledgeSearchResult>>(store.ReadDocumentContext(reference));
        Assert.All(context, chunk => Assert.Equal("text", chunk.Modality));
        Assert.Contains(context, chunk => chunk.Location?.Page == 1);
        Assert.Contains(context, chunk => chunk.Location?.Page == 2);
        Assert.True(context.Count < document.ChunkCount);
        Assert.All(context, chunk => Assert.Equal(Assert.Single(store.Read(chunk.Reference)).Text, chunk.Text));
    }

    [Fact]
    public async Task DocumentContextPreservesChangedNumbersPunctuationAndUniqueOcrContent()
    {
        const string paragraph = "Aster demonstrates clear reasoning and explains each conclusion carefully. Result 4.";
        var extraction = new KnowledgeExtraction { Segments = [
            new(paragraph, new() { Page = 1 }, "text"),
            new(paragraph.Replace("4", "3"), new() { Page = 1 }, "ocr"),
            new(paragraph.Replace("4", "-4"), new() { Page = 1 }, "ocr"),
            new(paragraph + " Additional handwritten note.", new() { Page = 1 }, "ocr"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)));
        var collection = await store.CreateCollectionAsync("Complementary evidence");
        var document = await Import(store, collection.Id, "review.pdf", "fixture");
        string reference = $"kb://{collection.Id}/{document.Id}@{document.Version}/0";
        var context = Assert.IsAssignableFrom<IReadOnlyList<KnowledgeSearchResult>>(store.ReadDocumentContext(reference));
        Assert.Equal(4, context.Count);
        Assert.Contains(context, chunk => chunk.Text.Contains("Result 3"));
        Assert.Contains(context, chunk => chunk.Text.Contains("Result -4"));
        Assert.Contains(context, chunk => chunk.Text.Contains("Additional handwritten note"));
    }

    [Fact]
    public async Task DocumentContextDropsExactOcrExcerptsSpanningNativeChunksButKeepsUniqueNotes()
    {
        const string passage = "The reviewer recommends clear written explanations, independent checking and careful revision before submission.";
        string native = new string('a', 90) + " " + passage + " " + new string('b', 80);
        var extraction = new KnowledgeExtraction { Segments = [
            new(native, new() { Page = 1 }, "text"),
            new("A distinct image caption.", new() { Page = 1 }, "ocr"),
            new(passage.Replace(" ", "\n  "), new() { Page = 1 }, "ocr"),
            new("Unique handwritten follow-up.", new() { Page = 1 }, "ocr"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)),
            options: new() { ChunkCharacters = 150, ChunkOverlapCharacters = 20 });
        var collection = await store.CreateCollectionAsync("Exact excerpts");
        var document = await Import(store, collection.Id, "review.pdf", "fixture");
        string reference = $"kb://{collection.Id}/{document.Id}@{document.Version}/0";
        var context = Assert.IsAssignableFrom<IReadOnlyList<KnowledgeSearchResult>>(store.ReadDocumentContext(reference));
        var ocr = context.Where(chunk => chunk.Modality == "ocr").ToArray();
        Assert.Equal(2, ocr.Length);
        Assert.Contains(ocr, chunk => chunk.Text == "A distinct image caption.");
        Assert.Contains(ocr, chunk => chunk.Text == "Unique handwritten follow-up.");
        Assert.All(context, chunk => Assert.Equal(Assert.Single(store.Read(chunk.Reference)).Text, chunk.Text));
    }

    [Fact]
    public async Task DocumentContextDoesNotDiscardDifferentWordOrNumericBoundaryMeanings()
    {
        var extraction = new KnowledgeExtraction { Segments = [
            new("An inadequate score: 14.5 or -4.5.", new() { Page = 1 }, "text"),
            new("adequate", new() { Page = 1 }, "ocr"),
            new("4.5", new() { Page = 1 }, "ocr"),
            new("4 5", new() { Page = 1 }, "ocr"),
        ] };
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(extraction)));
        var collection = await store.CreateCollectionAsync("Meaningful boundaries");
        var document = await Import(store, collection.Id, "review.pdf", "fixture");
        string reference = $"kb://{collection.Id}/{document.Id}@{document.Version}/0";
        var context = Assert.IsAssignableFrom<IReadOnlyList<KnowledgeSearchResult>>(store.ReadDocumentContext(reference));
        Assert.Equal(4, context.Count);
    }

    [Fact]
    public async Task DocumentContextEnforcesScopeVersionAndItsSeparateBudgetCeiling()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor(), options: new()
            { MaxContextCharacters = 32, MaxDocumentContextCharacters = 128 });
        var collection = await store.CreateCollectionAsync("Allowed");
        var other = await store.CreateCollectionAsync("Other");
        var document = await Import(store, collection.Id, "a.txt", new string('a', 80));
        string reference = $"kb://{collection.Id}/{document.Id}@{document.Version}/0";
        Assert.Equal(80, Assert.Single(store.ReadDocumentContext(reference, 128)!).Text.Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => store.Read(reference, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.ReadDocumentContext(reference, 129));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.ReadDocumentContext(reference, 0));
        Assert.Throws<KeyNotFoundException>(() => store.ReadDocumentContext(reference, 128, [other.Id]));
        Assert.Throws<KeyNotFoundException>(() => store.ReadDocumentContext(reference, 128, []));
        await Import(store, collection.Id, "a.txt", "replacement");
        Assert.Throws<KnowledgeStaleReferenceException>(() => store.ReadDocumentContext(reference, 128));
    }

    [Theory]
    [InlineData("zero")]
    [InlineData("dimensions")]
    public async Task CorruptStoredEmbeddingsFailBeforeIndexPublication(string corruption)
    {
        using (var store = new KnowledgeStore(_root, new TextExtractor(), new VectorProvider(),
            new() { ChunkCharacters = 16, ChunkOverlapCharacters = 0 }))
        {
            var collection = await store.CreateCollectionAsync("Corrupt vectors");
            await Import(store, collection.Id, "a.txt", "automobile maintenance instructions are here");
        }
        string path = Assert.Single(Directory.GetFiles(Path.Combine(_root, "documents")));
        var manifest = JsonNode.Parse(File.ReadAllText(path))!;
        manifest["Chunks"]![0]!["Vector"] = corruption == "zero" ? JsonNode.Parse("[0,0]") : JsonNode.Parse("[1,0,0]");
        File.WriteAllText(path, manifest.ToJsonString());
        Assert.Throws<InvalidDataException>(() => new KnowledgeStore(_root, new TextExtractor(), options: new() { ChunkCharacters = 16, ChunkOverlapCharacters = 0 }));
    }

    [Fact]
    public async Task HybridDoesNotBoostUnrelatedDocumentsSolelyForQueryFunctionWords()
    {
        using var store = new KnowledgeStore(_root, new TextExtractor(), new VectorProvider());
        var collection = await store.CreateCollectionAsync("Relevance");
        await Import(store, collection.Id, "a-noise.txt", "the apple is in the orchard and should be picked");
        await Import(store, collection.Id, "z-target.txt", "automobile maintenance");
        var result = await Search(store, collection.Id, "when should the car be serviced");
        Assert.Equal("z-target.txt", result[0].FileName);
        Assert.Equal("a-noise.txt", (await Search(store, collection.Id, "the in should"))[0].FileName);
    }

    [Fact]
    public async Task OversizedProvenanceCannotAmplifyChunkManifests()
    {
        using var store = new KnowledgeStore(_root, new DelegateExtractor((_, _) => Task.FromResult(new KnowledgeExtraction
            { Segments = [new("text", new() { Section = new string('x', 1001) })] })));
        var collection = await store.CreateCollectionAsync("Metadata bounds");
        await Assert.ThrowsAsync<InvalidDataException>(() => Import(store, collection.Id, "a.txt", "fixture"));
        Assert.Empty(store.ListDocuments(collection.Id));
    }

    private static async Task<KnowledgeDocument> Import(KnowledgeStore store, string collection, string name, string text, CancellationToken ct = default)
    {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(text));
        return await store.ImportAsync(collection, name, stream, "text/plain", ct);
    }
    private static Task<IReadOnlyList<KnowledgeSearchResult>> Search(KnowledgeStore store, string collection, string query) =>
        store.SearchAsync(query, new() { CollectionIds = [collection] });

    private sealed class TextExtractor : IKnowledgeExtractor
    {
        public int Calls;
        public bool Fail;
        public async Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken = default)
        {
            Calls++;
            if (Fail) throw new InvalidDataException("Intentional extraction failure.");
            return new() { ExtractorId = "text-v1", Segments = [new(await File.ReadAllTextAsync(context.FilePath, cancellationToken))] };
        }
    }
    private sealed class DelegateExtractor(Func<KnowledgeExtractionContext, CancellationToken, Task<KnowledgeExtraction>> extract) : IKnowledgeExtractor
    {
        public Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken = default) => extract(context, cancellationToken);
    }
    private class VectorProvider(string modelId = "test-space") : IKnowledgeEmbeddingProvider, IDisposable
    {
        public string ModelId => modelId;
        public int Dimensions => 2;
        public int Calls;
        public bool Disposed;
        public virtual Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, bool isQuery, CancellationToken cancellationToken = default)
        {
            Calls++;
            return Task.FromResult(texts.Select(t => t.Contains("car") || t.Contains("automobile") ? new float[] { 3, 0 } : new float[] { 0, 5 }).ToArray());
        }
        public void Dispose() => Disposed = true;
    }
    private sealed class InvalidVectorProvider(string invalid) : VectorProvider
    {
        public override Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, bool isQuery, CancellationToken cancellationToken = default) => Task.FromResult(invalid switch
        {
            "count" => Array.Empty<float[]>(), "dimension" => new[] { new[] { 1f } },
            "nan" => new[] { new[] { float.NaN, 0f } }, _ => new[] { new[] { 0f, 0f } },
        });
    }
    private sealed class BlockingQueryProvider : VectorProvider
    {
        public TaskCompletionSource Entered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public override async Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, bool isQuery, CancellationToken cancellationToken = default)
        {
            if (isQuery) { Entered.SetResult(); await Release.Task.WaitAsync(cancellationToken); }
            Assert.False(Disposed);
            return await base.EmbedAsync(texts, isQuery, cancellationToken);
        }
    }
}
