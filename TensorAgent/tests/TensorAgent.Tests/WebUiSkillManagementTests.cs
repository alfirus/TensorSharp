// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    // The real menu opens the real skills sheet; the tiny DOM fixture otherwise has
    // no static menu buttons. Its roster includes a bundled and a user-installed
    // skill, both of which TensorAgent can remove through its configured registry.
    private const string ManagedSkills = """
        var managedSettings = {
          thinkByDefault: false, defaultSkills: ['documents'], skillsEnabled: true, maxTokens: 2048
        };
        var managedCatalog = { enabled: true, installable: true, skills: [
          { id: 'documents', name: 'documents', description: 'Create documents.', origin: 'discovered' },
          { id: 'research', name: 'research', description: 'Research a topic.', origin: 'installed' }
        ] };
        R['/api/agent/settings'] = function (call) {
          if (call.method === 'POST') {
            managedSettings = call.body;
            managedCatalog.enabled = managedSettings.skillsEnabled;
          }
          return managedSettings;
        };
        R['/api/skills'] = function () { return managedCatalog; };
        ['documents', 'research'].forEach(function (name) {
          R['/api/skills/' + name] = function (call) {
            if (call.method !== 'DELETE') throw new Error('Expected DELETE');
            managedCatalog.skills = managedCatalog.skills.filter(function (s) { return s.name !== name; });
            return { removed: true, name: name };
          };
        });
        var skillsNav = document.createElement('button');
        skillsNav.className = 'opt';
        skillsNav.setAttribute('data-sheet', 'skills-sheet');
        document.getElementById('nav-sheet').appendChild(skillsNav);
        function managedSkill(name) {
          return document.getElementById('skills-list').querySelectorAll('.skillrow')
            .filter(function (row) { return row.getAttribute('data-skill') === name; })[0];
        }
        function openManagedSkills() { skillsNav.dispatch('click'); return settle(20); }
        """;

    [WebJavaScriptFact]
    public void SkillListSwitchesChooseTheNextMessagesSkillsWithoutOpeningDetailsOrFetchingAgain()
    {
        JsonElement result = Run(ManagedSkills, """
            return openManagedSkills().then(function () {
              var reads = __page.requests('/api/skills').length;
              var documents = managedSkill('documents').querySelector('.skill-toggle');
              var research = managedSkill('research').querySelector('.skill-toggle');
              var initial = { documents: documents.checked, research: research.checked,
                documentsLabel: documents.getAttribute('aria-label'), researchLabel: research.getAttribute('aria-label') };
              documents.checked = false;
              documents.dispatch('change');
              research = managedSkill('research').querySelector('.skill-toggle');
              research.checked = true;
              research.dispatch('change');
              __page.byId['text'].value = 'Research this subject';
              __page.byId['send'].dispatch('click');
              return { initial: initial, sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                extraReads: __page.requests('/api/skills').length - reads,
                open: __page.byId['skills-sheet'].classList.contains('on'),
                details: __page.byId['skill-sheet'].classList.contains('on') };
            });
            """);

        Assert.True(result.GetProperty("initial").GetProperty("documents").GetBoolean());
        Assert.False(result.GetProperty("initial").GetProperty("research").GetBoolean());
        Assert.Contains("documents", result.GetProperty("initial").GetProperty("documentsLabel").GetString());
        Assert.Contains("research", result.GetProperty("initial").GetProperty("researchLabel").GetString());
        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "research" }, Strings(request, "skills"));
        Assert.Equal(0, result.GetProperty("extraReads").GetInt32());
        Assert.True(result.GetProperty("open").GetBoolean());
        Assert.False(result.GetProperty("details").GetBoolean());
    }

    [WebJavaScriptFact]
    public void DisablingTheLastSkillInTheListSendsAnExplicitEmptySelection()
    {
        JsonElement result = Run(ManagedSkills, """
            return openManagedSkills().then(function () {
              var box = managedSkill('documents').querySelector('.skill-toggle');
              box.checked = false;
              box.dispatch('change');
              __page.byId['text'].value = 'Make a document';
              __page.byId['send'].dispatch('click');
              return { sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                chips: __page.byId['skillchips'].children.length };
            });
            """);

        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Empty(request.GetProperty("skills").EnumerateArray());
        Assert.False(request.GetProperty("skills_discovery").GetBoolean());
        Assert.Equal(0, result.GetProperty("chips").GetInt32());
    }

    [WebJavaScriptTheory]
    [InlineData("documents")]
    [InlineData("research")]
    public void RemovingBundledOrInstalledSkillsKeepsTheListOpenAndUpdatesSelection(string skill)
    {
        JsonElement result = Run(ManagedSkills, $$"""
            window.TensorAgent.setSkills(['documents', 'research']);
            return openManagedSkills().then(function () {
              managedSkill('{{skill}}').querySelector('.skill-remove').dispatch('click');
              return settle(20);
            }).then(function () {
              __page.byId['text'].value = 'Continue';
              __page.byId['send'].dispatch('click');
              return { present: !!managedSkill('{{skill}}'),
                remainingRows: __page.byId['skills-list'].querySelectorAll('.skillrow').length,
                deletes: __page.requests('/api/skills/{{skill}}').filter(function (c) { return c.method === 'DELETE'; }).length,
                sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                open: __page.byId['skills-sheet'].classList.contains('on'),
                details: __page.byId['skill-sheet'].classList.contains('on') };
            });
            """);

        Assert.False(result.GetProperty("present").GetBoolean());
        Assert.Equal(1, result.GetProperty("remainingRows").GetInt32());
        Assert.Equal(1, result.GetProperty("deletes").GetInt32());
        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { skill == "documents" ? "research" : "documents" }, Strings(request, "skills"));
        Assert.True(result.GetProperty("open").GetBoolean());
        Assert.False(result.GetProperty("details").GetBoolean());
    }

    [WebJavaScriptFact]
    public void FailedInlineRemovalKeepsTheSkillSelectedAndShowsTheServerErrorInTheList()
    {
        JsonElement result = Run(ManagedSkills + """
            R['/api/skills/documents'] = { __status: 403, body: { error: 'This skill cannot be removed.' } };
            """, """
            return openManagedSkills().then(function () {
              managedSkill('documents').querySelector('.skill-remove').dispatch('click');
              return settle(20);
            }).then(function () {
              var row = managedSkill('documents');
              __page.byId['text'].value = 'Make a document';
              __page.byId['send'].dispatch('click');
              return { present: !!row, selected: row.querySelector('.skill-toggle').checked,
                removeEnabled: !row.querySelector('.skill-remove').disabled,
                sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                error: __page.byId['skills-error'].textContent,
                open: __page.byId['skills-sheet'].classList.contains('on') };
            });
            """);

        Assert.True(result.GetProperty("present").GetBoolean());
        Assert.True(result.GetProperty("selected").GetBoolean());
        Assert.True(result.GetProperty("removeEnabled").GetBoolean());
        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "documents" }, Strings(request, "skills"));
        Assert.Contains("This skill cannot be removed.", result.GetProperty("error").GetString());
        Assert.True(result.GetProperty("open").GetBoolean());
    }

    [WebJavaScriptFact]
    public void SkillsDisabledGloballyStillAllowsRemovalInTheListButCannotSelectSkills()
    {
        JsonElement result = Run(ManagedSkills + """
            managedSettings.skillsEnabled = false;
            managedCatalog.enabled = false;
            """, """
            return openManagedSkills().then(function () {
              var row = managedSkill('research');
              var box = row.querySelector('.skill-toggle');
              var before = { selectionDisabled: box.disabled, removeEnabled: !row.querySelector('.skill-remove').disabled };
              box.checked = true;
              box.dispatch('change');
              managedSkill('research').querySelector('.skill-remove').dispatch('click');
              return settle(20).then(function () {
                __page.byId['text'].value = 'Continue without skills';
                __page.byId['send'].dispatch('click');
                return { before: before, removed: !managedSkill('research'),
                  off: __page.byId['skills-list'].classList.contains('off'),
                  sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                  open: __page.byId['skills-sheet'].classList.contains('on') };
              });
            });
            """);

        Assert.True(result.GetProperty("before").GetProperty("selectionDisabled").GetBoolean());
        Assert.True(result.GetProperty("before").GetProperty("removeEnabled").GetBoolean());
        Assert.True(result.GetProperty("removed").GetBoolean());
        Assert.True(result.GetProperty("off").GetBoolean());
        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.False(request.TryGetProperty("skills", out _));
        Assert.False(request.GetProperty("skills_discovery").GetBoolean());
        Assert.True(result.GetProperty("open").GetBoolean());
    }

    [WebJavaScriptFact]
    public void InlineRemovalDisablesThePendingRowAndDoesNotSubmitDuplicateDeletes()
    {
        JsonElement result = Run(ManagedSkills + Held, """
            return openManagedSkills().then(function () {
              hold('/api/skills/research');
              var row = managedSkill('research');
              var remove = row.querySelector('.skill-remove');
              remove.dispatch('click');
              remove.dispatch('click');
              row = managedSkill('research');
              var pending = { removeDisabled: row.querySelector('.skill-remove').disabled,
                selectionDisabled: row.querySelector('.skill-toggle').disabled };
              return settle(10).then(function () {
                pending.deletes = heldFor('/api/skills/research').length;
                managedCatalog.skills = managedCatalog.skills.filter(function (s) { return s.name !== 'research'; });
                heldFor('/api/skills/research')[0].answer({ removed: true, name: 'research' });
                return settle(20).then(function () {
                  return { pending: pending, removed: !managedSkill('research'),
                    otherEnabled: !managedSkill('documents').querySelector('.skill-toggle').disabled };
                });
              });
            });
            """);

        Assert.True(result.GetProperty("pending").GetProperty("removeDisabled").GetBoolean());
        Assert.True(result.GetProperty("pending").GetProperty("selectionDisabled").GetBoolean());
        Assert.Equal(1, result.GetProperty("pending").GetProperty("deletes").GetInt32());
        Assert.True(result.GetProperty("removed").GetBoolean());
        Assert.True(result.GetProperty("otherEnabled").GetBoolean());
    }

    [WebJavaScriptFact]
    public void ASlowRosterResponseCannotRestoreASkillRemovedFromTheOpenList()
    {
        JsonElement result = Run(ManagedSkills + Held, """
            return openManagedSkills().then(function () {
              hold('/api/skills');
              skillsNav.dispatch('click');
              return settle(10);
            }).then(function () {
              var staleCatalog = JSON.parse(JSON.stringify(managedCatalog));
              managedSkill('documents').querySelector('.skill-remove').dispatch('click');
              return settle(20).then(function () {
                heldFor('/api/skills').forEach(function (h) { h.answer(staleCatalog); });
                return settle(20);
              });
            }).then(function () {
              return { present: !!managedSkill('documents'),
                remainingRows: __page.byId['skills-list'].querySelectorAll('.skillrow').length,
                open: __page.byId['skills-sheet'].classList.contains('on') };
            });
            """);

        Assert.False(result.GetProperty("present").GetBoolean());
        Assert.Equal(1, result.GetProperty("remainingRows").GetInt32());
        Assert.True(result.GetProperty("open").GetBoolean());
    }

    [WebJavaScriptFact]
    public void FailedMasterSwitchSaveRestoresItsSelectionAndPreventsOverlappingSaves()
    {
        JsonElement result = Run(ManagedSkills + Held, """
            return openManagedSkills().then(function () {
              hold('/api/agent/settings');
              var master = __page.byId['skills-master'];
              master.checked = false;
              master.dispatch('change');
              var pendingDisabled = master.disabled;
              master.checked = true;
              master.dispatch('change');
              return settle(10).then(function () {
                var saves = heldFor('/api/agent/settings').length;
                heldFor('/api/agent/settings')[0].answer({ __status: 500, body: { error: 'Settings disk is full.' } });
                return settle(20).then(function () {
                  __page.byId['text'].value = 'Make a document';
                  __page.byId['send'].dispatch('click');
                  return { pendingDisabled: pendingDisabled, saves: saves,
                    enabled: master.checked, ready: !master.disabled,
                    selected: managedSkill('documents').querySelector('.skill-toggle').checked,
                    sent: __page.requests('/api/chat').map(function (c) { return c.body; }),
                    error: __page.byId['skills-error'].textContent };
                });
              });
            });
            """);

        Assert.True(result.GetProperty("pendingDisabled").GetBoolean());
        Assert.Equal(1, result.GetProperty("saves").GetInt32());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.True(result.GetProperty("ready").GetBoolean());
        Assert.True(result.GetProperty("selected").GetBoolean());
        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Equal(new[] { "documents" }, Strings(request, "skills"));
        Assert.Contains("500", result.GetProperty("error").GetString());
    }

    [WebJavaScriptTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void ChangingTheMasterSwitchWhileTheFirstRosterLoadsStillShowsTheRoster(bool saveSucceeds)
    {
        JsonElement result = Run(ManagedSkills + Held, $$"""
            hold('/api/skills');
            var initialCatalog = JSON.parse(JSON.stringify(managedCatalog));
            skillsNav.dispatch('click');
            return settle(10).then(function () {
              if (!{{(saveSucceeds ? "true" : "false")}}) {
                R['/api/agent/settings'] = { __status: 500, body: { error: 'Could not save settings.' } };
              }
              var master = __page.byId['skills-master'];
              master.checked = false;
              master.dispatch('change');
              return settle(20);
            }).then(function () {
              var beforeRoster = { master: __page.byId['skills-master'].checked,
                chips: __page.byId['skillchips'].textContent };
              heldFor('/api/skills')[0].answer(initialCatalog);
              return settle(20).then(function () {
                return { beforeRoster: beforeRoster,
                  rows: __page.byId['skills-list'].querySelectorAll('.skillrow').length,
                  master: __page.byId['skills-master'].checked,
                  selected: managedSkill('documents').querySelector('.skill-toggle').checked };
              });
            });
            """);

        Assert.Equal(!saveSucceeds, result.GetProperty("beforeRoster").GetProperty("master").GetBoolean());
        string chips = result.GetProperty("beforeRoster").GetProperty("chips").GetString()!;
        if (saveSucceeds)
            Assert.Equal(string.Empty, chips);
        else
            Assert.Contains("documents", chips, StringComparison.Ordinal);
        Assert.Equal(2, result.GetProperty("rows").GetInt32());
        Assert.Equal(!saveSucceeds, result.GetProperty("master").GetBoolean());
        Assert.Equal(!saveSucceeds, result.GetProperty("selected").GetBoolean());
    }

    [WebJavaScriptFact]
    public void ARemovingSkillsDetailsAreDisabledAndStaleControlsCannotSelectItAfterRemoval()
    {
        JsonElement result = Run(ManagedSkills + Held, """
            return openManagedSkills().then(function () {
              managedSkill('documents').querySelector('.skill-info').dispatch('click');
              __page.byId['sheet-bg'].dispatch('click');
              return openManagedSkills();
            }).then(function () {
              hold('/api/skills/documents');
              var row = managedSkill('documents');
              var info = row.querySelector('.skill-info');
              var staleRowSwitch = row.querySelector('.skill-toggle');
              row.querySelector('.skill-remove').dispatch('click');
              var pendingInfoDisabled = info.disabled;
              info.dispatch('click');
              var openedWhileRemoving = __page.byId['skill-sheet'].classList.contains('on');
              return settle(10).then(function () {
                managedCatalog.skills = managedCatalog.skills.filter(function (s) { return s.name !== 'documents'; });
                heldFor('/api/skills/documents')[0].answer({ removed: true, name: 'documents' });
                return settle(20);
              }).then(function () {
                staleRowSwitch.checked = true;
                staleRowSwitch.dispatch('change');
                __page.byId['skill-on'].checked = true;
                __page.byId['skill-on'].dispatch('change');
                info.dispatch('click');
                __page.byId['text'].value = 'Continue without the removed skill';
                __page.byId['send'].dispatch('click');
                return { pendingInfoDisabled: pendingInfoDisabled, openedWhileRemoving: openedWhileRemoving,
                  present: !!managedSkill('documents'),
                  openedAfterRemoving: __page.byId['skill-sheet'].classList.contains('on'),
                  detailSwitchDisabled: __page.byId['skill-on'].disabled,
                  sent: __page.requests('/api/chat').map(function (c) { return c.body; }) };
              });
            });
            """);

        Assert.True(result.GetProperty("pendingInfoDisabled").GetBoolean());
        Assert.False(result.GetProperty("openedWhileRemoving").GetBoolean());
        Assert.False(result.GetProperty("present").GetBoolean());
        Assert.False(result.GetProperty("openedAfterRemoving").GetBoolean());
        Assert.True(result.GetProperty("detailSwitchDisabled").GetBoolean());
        JsonElement request = Assert.Single(result.GetProperty("sent").EnumerateArray());
        Assert.Empty(request.GetProperty("skills").EnumerateArray());
        Assert.False(request.GetProperty("skills_discovery").GetBoolean());
    }
}
