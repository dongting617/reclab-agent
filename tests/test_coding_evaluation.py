"""Coding benchmark contracts; ordinary tests do not execute candidate code."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reclab_agent.evaluation.coding import check_rows,run_coding_suite
from reclab_agent.experiments.workspace import Workspace
from reclab_agent.tools.coding import CodingToolbox
from reclab_agent.execution.coding import CodingContainer
from reclab_agent.domain.configuration import read_json_strict


class CodingEvaluationTests(unittest.TestCase):
    def test_tools_confine_patches_and_bound_public_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'src').mkdir();(root/'src/solution.py').write_text('def solve(x): return x\n')
            workspace=Workspace(root,['src/solution.py'])
            tools=CodingToolbox(workspace,lambda:{'passed':1},max_patches=1,max_checks=1)
            self.assertFalse(tools.invoke('read_source',{'path':'../tasks/coding.json'})['ok'])
            old=workspace.read('src/solution.py')
            self.assertTrue(tools.invoke('apply_patch',{'path':'src/solution.py','old':'return x','new':'return x + 1','expected_sha256':old['sha256']})['ok'])
            self.assertEqual(tools.invoke('apply_patch',{})['error']['code'],'INVALID_ARGUMENTS')
            self.assertTrue(tools.invoke('run_checks')['ok'])
            self.assertEqual(tools.invoke('run_checks')['error']['code'],'BUDGET_EXCEEDED')

    def test_hidden_grading_does_not_coerce_boolean_item_ids(self):
        class Executor:
            def run(self,*args):return {'outputs':[[True]],'error':None,'exit_code':0,'elapsed_ms':1}
        result=check_rows(Executor(),Path('.'),[{'input':{},'expected':[1]}])
        self.assertEqual(result['passed'],0)

    def test_mutated_manifest_rejected_before_container_execution(self):
        source=Path(__file__).resolve().parents[1]/'tasks'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            data=json.loads((source/'coding.json').read_text())
            data['cases'][0]['description']='changed after freezing'
            (root/'coding.json').write_text(json.dumps(data))
            (root/'coding-freeze.json').write_bytes((source/'coding-freeze.json').read_bytes())
            executor=type('Executor',(),{'image':'sha256:'+'a'*64})()
            with self.assertRaisesRegex(ValueError,'changed after freezing'):
                run_coding_suite(root/'coding.json',root/'report.json',executor=executor)

    def test_image_tag_is_not_an_immutable_execution_identity(self):
        with self.assertRaisesRegex(ValueError,'immutable image'):
            CodingContainer('python:latest')

    def test_interrupted_request_is_charged_and_trial_is_not_replayed(self):
        class InterruptedBackend:
            model='offline-interruption-test'
            def respond(self,**kwargs):
                raise KeyboardInterrupt('controlled controller interruption')
        manifest=Path(__file__).resolve().parents[1]/'tasks/coding.json'
        executor=type('Executor',(),{'image':'sha256:'+'a'*64})()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'report.json'
            arguments=dict(executor=executor,mode='react',backend=InterruptedBackend(),max_steps=1,max_requests=1)
            with self.assertRaises(KeyboardInterrupt):
                run_coding_suite(manifest,output,**arguments)
            self.assertEqual(read_json_strict(output)['model_requests'],1)
            report=run_coding_suite(manifest,output,**arguments)
            self.assertEqual(report['model_requests'],1)
            self.assertEqual(report['trials'],8)
            self.assertEqual(report['results'][0]['status'],'interrupted')
            self.assertTrue(all(row['status']=='not_run' for row in report['results'][1:]))
            self.assertFalse(report['resource_usage']['usage_complete'])

    def test_actual_evaluation_cli_disables_unaccounted_sdk_retries(self):
        from reclab_agent.cli.evaluate import main
        summary=dict(kind='offline-test',status='completed',tasks=0,passed=0,
                     pass_rate=0,model_requests=0,resource_usage={})
        with patch('reclab_agent.core.OpenAIResponsesBackend') as backend, \
             patch('reclab_agent.evaluation.diagnostics.run_diagnostic_suite',return_value=summary), \
             redirect_stdout(io.StringIO()):
            client=backend.return_value.client
            self.assertEqual(main(['diagnostics','--mode','react','--model','offline-test']),0)
            client.with_options.assert_called_once_with(max_retries=0,timeout=40)
