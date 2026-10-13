using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void ImportedAndReindexedImageNeedsExplicitChatSelectionAndNearbyControlSelectsItsCollection()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method === 'POST') {
                libraryDocuments = [{ id: 'doc1', collectionId: 'alpha', fileName: 'banner_1.png',
                  version: 1, chunkCount: 1, status: 'Ready', warnings: [] }];
                return { documents: libraryDocuments, errors: [] };
              }
              return { documents: libraryDocuments };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-file-input'];
              input.files = [{ name: 'banner_1.png', size: 1024, type: 'image/png' }];
              input.dispatch('change', { target: input });
              return settle(25);
            }).then(function () {
              __page.byId['knowledge-documents'].querySelector('.knowledge-reindex').dispatch('click');
              return settle(25);
            }).then(function () {
              var before = { checked: __page.byId['knowledge-use-active'].checked,
                help: __page.byId['knowledge-scope-help'].textContent, chips: __page.byId['knowledgechips'].children.length };
              __page.byId['knowledge-use-active'].checked = true;
              __page.byId['knowledge-use-active'].dispatch('change');
              __page.byId['text'].value = '请到知识库中寻找tensorsharp的宣传画';
              __page.byId['send'].dispatch('click');
              return { before: before, label: __page.byId['knowledge-use-active-label'].textContent,
                after: __page.byId['knowledge-scope-help'].textContent,
                rowChecked: knowledgeCollection('alpha').querySelector('.knowledge-toggle').checked,
                selected: __page.requests('/api/chat')[0].body.knowledgeCollections,
                source: __page.byId['knowledge-documents'].querySelector('.knowledge-source').href };
            });
            """);
        Assert.False(result.GetProperty("before").GetProperty("checked").GetBoolean());
        Assert.Equal(0, result.GetProperty("before").GetProperty("chips").GetInt32());
        Assert.Contains("Select this collection", result.GetProperty("before").GetProperty("help").GetString());
        Assert.Equal("Use Research in this chat", result.GetProperty("label").GetString());
        Assert.Contains("selected for this chat", result.GetProperty("after").GetString());
        Assert.True(result.GetProperty("rowChecked").GetBoolean());
        Assert.Equal(new[] { "alpha" }, Strings(result, "selected"));
        Assert.EndsWith("version=2", result.GetProperty("source").GetString());
    }

    [WebJavaScriptFact]
    public void CollectionChatControlTracksBrowsingAndCheckboxChangesWithoutChangingAnotherCollectionsSelection()
    {
        JsonElement result = Run(KnowledgeLibrary, """
            return openKnowledgeLibrary().then(function () {
              chooseKnowledge('alpha', true);
              var selected = __page.byId['knowledge-use-active'].checked;
              knowledgeCollection('beta').querySelector('.knowledge-browse').dispatch('click');
              return settle(20).then(function () {
                var browsing = { checked: __page.byId['knowledge-use-active'].checked,
                  label: __page.byId['knowledge-use-active-label'].textContent };
                __page.byId['knowledge-use-active'].checked = true;
                __page.byId['knowledge-use-active'].dispatch('change');
                chooseKnowledge('beta', false);
                var unselected = __page.byId['knowledge-use-active'].checked;
                __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
                return { selected: selected, browsing: browsing, unselected: unselected,
                  scope: __page.requests('/api/chat')[0].body.knowledgeCollections };
              });
            });
            """);
        Assert.True(result.GetProperty("selected").GetBoolean());
        Assert.False(result.GetProperty("browsing").GetProperty("checked").GetBoolean());
        Assert.Equal("Use Private notes in this chat", result.GetProperty("browsing").GetProperty("label").GetString());
        Assert.False(result.GetProperty("unselected").GetBoolean());
        Assert.Equal(new[] { "alpha" }, Strings(result, "scope"));
    }

    [WebJavaScriptFact]
    public void NearbyCollectionChatControlHonorsSelectionLimitAndRestoresItsUncheckedState()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            libraryCollections = Array.from({ length: 33 }, function (_, index) {
              return { id: 'collection' + index, name: 'Collection ' + index };
            });
            R['/api/knowledge/collections/collection0/documents'] = { documents: [] };
            R['/api/knowledge/collections/collection32/documents'] = { documents: [] };
            """, """
            return openKnowledgeLibrary().then(function () {
              libraryCollections.slice(0, 32).forEach(function (collection) { chooseKnowledge(collection.id, true); });
              knowledgeCollection('collection32').querySelector('.knowledge-browse').dispatch('click');
              return settle(20).then(function () {
                __page.byId['knowledge-use-active'].checked = true;
                __page.byId['knowledge-use-active'].dispatch('change');
                __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
                return { checked: __page.byId['knowledge-use-active'].checked,
                  rowChecked: knowledgeCollection('collection32').querySelector('.knowledge-toggle').checked,
                  error: __page.byId['knowledge-error'].textContent,
                  scope: __page.requests('/api/chat')[0].body.knowledgeCollections };
              });
            });
            """);
        Assert.False(result.GetProperty("checked").GetBoolean());
        Assert.False(result.GetProperty("rowChecked").GetBoolean());
        Assert.Contains("at most 32", result.GetProperty("error").GetString());
        Assert.Equal(32, result.GetProperty("scope").GetArrayLength());
        Assert.DoesNotContain("collection32", Strings(result, "scope"));
    }
}
