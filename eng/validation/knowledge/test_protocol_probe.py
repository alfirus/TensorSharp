"""Hermetic protocol-observer regression tests; no host/model correctness claim."""
import json
import unittest
from protocol_probe import parse_reply, protocol_body, chinese_phrase_checks


def sse(events):
    return ''.join('data: '+json.dumps(event)+'\n\n' for event in events).encode()


class ProtocolObserverTests(unittest.TestCase):
    def test_chinese_ocr_spacing_is_reported_without_accepting_wrong_characters(self):
        phrases=['竹林','蓝色','二十七']
        spaced=chinese_phrase_checks('竹林 项目：蓝 色 箱 子 有 二 十 七 个。',phrases)
        self.assertTrue(spaced['passed'])
        self.assertFalse(spaced['rawExact'])
        self.assertTrue(spaced['whitespaceNormalized'])
        self.assertTrue(chinese_phrase_checks('竹林项目：蓝色箱子有二十七个。',phrases)['rawExact'])
        wrong=chinese_phrase_checks('竹林 项目：蓝 色 箱 子 有 二 十 六 个。',phrases)
        self.assertFalse(wrong['passed'])
        self.assertNotIn('二十七',wrong['found'])

    def test_chat_accumulates_content_and_retains_terminal_sources(self):
        sources=[{'documentId':'doc'}]
        raw=sse([{'choices':[{'delta':{'content':'One '},'finish_reason':None}]},
            {'choices':[{'delta':{'content':'answer'},'finish_reason':None}]},
            {'choices':[{'delta':{},'finish_reason':'stop'}],'knowledge_sources':sources},
            {'choices':[],'usage':{'completion_tokens':2}}])+b'data: [DONE]\n\n'
        answer, terminal, _, complete=parse_reply('chat',raw,True)
        self.assertEqual('One answer',answer)
        self.assertEqual(sources,terminal['knowledge_sources'])
        self.assertTrue(complete)

    def test_responses_checks_terminal_content_matches_deltas(self):
        response={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'answer'}]}]}
        raw=sse([{'type':'response.output_text.delta','delta':'answer'},
            {'type':'response.completed','response':response}])
        self.assertEqual(('answer',response),parse_reply('responses',raw,True)[:2])
        bad=sse([{'type':'response.output_text.delta','delta':'partial'},
            {'type':'response.completed','response':response}])
        with self.assertRaises(AssertionError):
            parse_reply('responses',bad,True)

    def test_ollama_keeps_final_content_and_tool_trace(self):
        steps=[{'tool':'shell','ok':True,'output':'ROWS=500 TOTAL_CENTS=12'}]
        events=[{'message':{'content':'First '},'done':False},
            {'message':{'content':'last'},'done':True,'done_reason':'stop','knowledge_tool_steps':steps}]
        answer, terminal, _, complete=parse_reply('ollama','\n'.join(map(json.dumps,events)).encode(),True)
        self.assertEqual('First last',answer)
        self.assertEqual(steps,terminal['knowledge_tool_steps'])
        self.assertTrue(complete)

    def test_generate_does_not_mistake_length_termination_for_success(self):
        raw=json.dumps({'response':'partial','done':True,'done_reason':'length'}).encode()
        self.assertFalse(parse_reply('generate',raw,True)[3])

    def test_json_length_termination_never_passes(self):
        for protocol, value in (
            ('chat',{'choices':[{'message':{'content':'partial'},'finish_reason':'length'}]}),
            ('responses',{'status':'incomplete','output':[]}),
            ('ollama',{'message':{'content':'partial'},'done':True,'done_reason':'length'}),
            ('generate',{'response':'partial','done':True,'done_reason':'length'})):
            self.assertFalse(parse_reply(protocol,json.dumps(value).encode(),False)[3],protocol)

    def test_unfinished_stream_never_passes(self):
        self.assertFalse(parse_reply('chat',sse([{'choices':[{'delta':{'content':'looks complete'}}]}]),True)[3])

    def test_request_scope_and_tools_are_explicit_for_all_protocols(self):
        for protocol in ('chat','responses','ollama','generate'):
            body=protocol_body(protocol,'test','question',True,['scope'])
            self.assertEqual(['scope'],body['knowledgeCollections'])
            self.assertEqual('none',body['tool_choice'])
            self.assertNotIn('tool_choice',protocol_body(protocol,'test','question',False,['scope'],tools=True))


if __name__=='__main__':
    unittest.main()
