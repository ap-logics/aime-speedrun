import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from speedrun import Config, extract_answer, load_questions, run, pop_candidate
from export_questions import project


def result(answer, finish='stop'):
    return {'choices': [{'message': {'content': '<think>check</think>\\boxed{' + str(answer) + '}'},
                          'finish_reason': finish}], 'usage': {'completion_tokens': 10}}


class InputTests(unittest.TestCase):
    def test_agreement_priority_and_fifo_ties(self):
        a = ({'problem_idx': 1}, 0, '7', None)
        b = ({'problem_idx': 2}, 0, '8', None)
        c = ({'problem_idx': 3}, 0, '9', None)
        ready = deque([a,b,c])
        self.assertEqual(pop_candidate(ready, {(1,'7'):1,(2,'8'):2,(3,'9'):1}, 'agreement'), b)
        self.assertEqual(list(ready), [a,c])
        self.assertEqual(pop_candidate(ready, {(1,'7'):1,(3,'9'):1}, 'agreement'), a)

    def test_final_answer_only(self):
        self.assertIsNone(extract_answer('<think>\\boxed{7}', 'stop'))
        self.assertIsNone(extract_answer('\\boxed{7}', 'length'))
        self.assertEqual(extract_answer('<think>\\boxed{7}</think>\\boxed{042}', 'stop'), '42')
        self.assertIsNone(extract_answer('\\boxed{1000}', 'stop'))

    def test_projection_handles_nested_and_escaped_excluded_values(self):
        row = {'excluded': {'nested': ['a } , \\"', 12]}, 'problem_idx': 1,
               'problem': 'Synthetic \\"problem\\" and {braces}', 'answer': 'EXCLUDED_CANARY'}
        exported = project(json.dumps(row))
        self.assertEqual(exported, {key: row[key] for key in ('problem_idx', 'problem')})
        self.assertNotIn('EXCLUDED_CANARY', json.dumps(exported))

    def test_excluded_values_are_not_json_decoded(self):
        # Lexical projection must not interpret unselected values, even invalid JSON tokens.
        self.assertEqual(project('{"problem_idx":1,"answer":DO_NOT_DECODE,"problem":"test"}'),
                         {'problem_idx': 1, 'problem': 'test'})

    def test_solver_rejects_non_question_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'questions.jsonl'
            path.write_text('{"problem_idx":1,"problem":"Synthetic","answer":123}\n')
            with self.assertRaises(ValueError):
                load_questions(path)


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_samples_use_equal_budgets_and_distinct_seeds(self):
        requests, checked = [], []
        async def generate(payload):
            requests.append(payload)
            return result(7 if len(requests) == 1 else 9)
        async def verify(idx, answer):
            checked.append(answer)
            return {'verdict': answer == '9'}
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify,
                            Config(samples=2, max_tokens=100, retry_tokens=200), lambda e: None, time.monotonic())
        self.assertEqual([p['max_tokens'] for p in requests], [100, 100])
        self.assertEqual(len({p['seed'] for p in requests}), 2)
        self.assertEqual(checked, ['7','9'])
        self.assertEqual(summary['solved'], 1)

    async def test_verified_sample_cancels_its_sibling(self):
        sibling_started = asyncio.Event()
        calls = 0
        async def generate(payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                await sibling_started.wait()
                return result(7)
            sibling_started.set()
            await asyncio.sleep(20)
        async def verify(idx, answer):
            return {'verdict': True}
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify,
                            Config(samples=2), lambda e: None, time.monotonic())
        self.assertEqual(summary['solved'], 1)
        self.assertEqual(summary['counts']['sample_cancellations_requested'], 1)
        self.assertEqual(summary['counts']['cancelled_generation_requests'], 1)

    async def exercise_overlap(self, early_correct, complete_before_verdict=False):
        resumed, released = asyncio.Event(), asyncio.Event()
        checked, events = [], []
        async def generate(payload):
            if len(payload['messages']) == 1:
                return {'choices': [{'message': {'content': '<think>unfinished '},
                                     'finish_reason': 'length'}], 'usage': {'completion_tokens': 80}}
            if 'stop' in payload:
                return {'choices': [{'message': {'content': '7'}, 'finish_reason': 'stop',
                                     'stop_reason': '}'}], 'usage': {'completion_tokens': 1}}
            resumed.set()
            await released.wait()
            return {'choices': [{'message': {'content': 'finished</think>\\boxed{9}'},
                                 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10}}
        async def verify(idx, answer):
            checked.append(answer)
            if answer == '7':
                await asyncio.wait_for(resumed.wait(), 1)
                if not early_correct or complete_before_verdict:
                    released.set()
                if complete_before_verdict:
                    await asyncio.sleep(0.03)
                return {'verdict': early_correct}
            return {'verdict': True}
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify,
                            Config(max_tokens=100, probe_tokens=80, finalize_tokens=32,
                                   probe_policy='always', overlap_probe=1), events.append, time.monotonic())
        return summary, checked, events

    async def test_overlap_cancels_reasoning_after_early_success(self):
        summary, checked, events = await self.exercise_overlap(True)
        self.assertEqual(checked, ['7'])
        self.assertEqual(summary['counts']['cancelled_generation_requests'], 1)
        self.assertEqual(summary['solved'], 1)

    async def test_overlap_failure_uses_existing_continuation(self):
        summary, checked, events = await self.exercise_overlap(False)
        self.assertEqual(checked, ['7', '9'])
        self.assertEqual(summary['counts']['speculative_continuations_started'], 1)
        self.assertEqual(summary['completed_response_tokens'], 91)
        self.assertEqual(summary['solved'], 1)

    async def test_finished_speculation_is_not_checked_after_early_success(self):
        summary, checked, events = await self.exercise_overlap(True, complete_before_verdict=True)
        self.assertEqual(checked, ['7'])
        self.assertEqual(summary['completed_response_tokens'], 91)
        self.assertEqual(summary['solved'], 1)

    async def exercise_probe(self, policy, early_correct):
        requests, checked, events = [], [], []
        reasoning = '<think>unfinished runtime reasoning '
        async def generate(payload):
            requests.append(payload)
            if len(requests) == 1:
                return {'choices': [{'message': {'content': reasoning}, 'finish_reason': 'length'}],
                        'usage': {'completion_tokens': 80}}
            if 'stop' in payload:
                return {'choices': [{'message': {'content': '7'}, 'finish_reason': 'stop',
                                     'stop_reason': '}'}], 'usage': {'completion_tokens': 1}}
            self.assertEqual(payload['messages'][-1]['content'], reasoning)
            self.assertEqual(payload['max_tokens'], 20)
            return {'choices': [{'message': {'content': 'more work</think>\\boxed{9}'},
                                 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10}}
        async def verify(idx, answer):
            checked.append(answer)
            return {'verdict': early_correct if answer == '7' else True}
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify,
                            Config(max_tokens=100, probe_tokens=80, finalize_tokens=32,
                                   probe_policy=policy), events.append, time.monotonic())
        return summary, requests, checked, events

    async def test_settled_success_avoids_continuation(self):
        summary, requests, checked, events = await self.exercise_probe('always', True)
        self.assertEqual(len(requests), 2)
        self.assertEqual(checked, ['7'])
        self.assertEqual(summary['solved'], 1)
        self.assertEqual(summary['completed_response_tokens'], 81)

    async def test_settled_failure_resumes_original_reasoning(self):
        summary, requests, checked, events = await self.exercise_probe('always', False)
        self.assertEqual(checked, ['7', '9'])
        self.assertEqual(summary['completed_response_tokens'], 91)
        self.assertEqual(summary['counts']['continuations_after_failed_probe'], 1)
        self.assertEqual(summary['solved'], 1)

    async def test_pricing_skips_probe_when_continuation_is_cheaper(self):
        summary, requests, checked, events = await self.exercise_probe('priced', False)
        self.assertEqual(checked, ['9'])
        decision = next(e for e in events if e['event'] == 'probe_decision')
        self.assertFalse(decision['buy'])
        self.assertLess(decision['estimated_saved_s'], decision['estimated_wait_s'])
        self.assertEqual(summary['completed_response_tokens'], 91)

    async def test_finalization_keeps_runtime_reasoning_and_charges_tokens(self):
        requests = []
        checked = []
        async def generate(payload):
            requests.append(payload)
            if len(requests) == 1:
                return result(123, 'length')
            return {'choices': [{'message': {'content': '17'}, 'finish_reason': 'stop',
                                  'stop_reason': '}'}], 'usage': {'completion_tokens': 2}}
        async def verify(idx, answer):
            checked.append(answer)
            return {'verdict': True}
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify,
                            Config(finalize_tokens=32), lambda e: None, time.monotonic())
        self.assertEqual(checked, ['17'])
        self.assertEqual(summary['completed_response_tokens'], 12)
        self.assertTrue(requests[1]['continue_final_message'])
        self.assertFalse(requests[1]['add_generation_prompt'])
        self.assertTrue(requests[1]['messages'][-1]['content'].endswith('\\boxed{'))
        self.assertEqual(summary['counts']['generation_requests'], 2)

    async def test_generation_overlaps_serial_grading(self):
        q = [{'problem_idx': i, 'problem': str(i)} for i in range(1, 4)]
        events = []
        active_grade = 0
        max_grades = 0
        grade_started = asyncio.Event()
        others_done = asyncio.Event()
        other_count = 0

        async def generate(payload):
            nonlocal other_count
            idx = int(payload['messages'][0]['content'].split('\n')[0])
            if idx != 1:
                await grade_started.wait()
                other_count += 1
                if other_count == 2:
                    others_done.set()
            return result(idx)

        async def verify(idx, answer):
            nonlocal active_grade, max_grades
            active_grade += 1
            max_grades = max(max_grades, active_grade)
            grade_started.set()
            await others_done.wait()
            await asyncio.sleep(0.08)
            active_grade -= 1
            return {'verdict': True}

        summary = await run(q, generate, verify, Config(concurrency=3, stop_at=3), events.append, time.monotonic())
        self.assertEqual(summary['solved'], 3)
        self.assertEqual(max_grades, 1)
        self.assertGreaterEqual(summary['milestones_s']['3'], 0.24)
        submit = next(e for e in events if e['event'] == 'submit')
        verdict = next(e for e in events if e['event'] == 'verdict')
        between = events[events.index(submit) + 1:events.index(verdict)]
        self.assertTrue(any(e['event'] == 'generation' for e in between))

    async def test_truncation_escalates_and_rejected_duplicate_is_not_resubmitted(self):
        generated = []
        checked = []
        async def generate(payload):
            generated.append(payload['max_tokens'])
            return result(17, 'length' if len(generated) == 1 else 'stop')
        async def verify(idx, answer):
            checked.append((idx, answer))
            return {'verdict': False}
        cfg = Config(attempts=3, max_tokens=100, retry_tokens=200)
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify, cfg, lambda e: None, time.monotonic())
        self.assertEqual(generated, [100, 200, 200])
        self.assertEqual(checked, [(1, '17')])
        self.assertEqual(summary['counts']['duplicates_skipped'], 1)

    async def test_deadline_cancels_work(self):
        cancelled = False
        async def generate(payload):
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            finally:
                cancelled = True
        async def verify(idx, answer):
            self.fail('No candidate should be sent')
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify,
                            Config(deadline_s=0.03), lambda e: None, time.monotonic())
        self.assertTrue(cancelled)
        self.assertEqual(summary['outcome'], 'deadline')
        self.assertIsNone(summary['time_to_18_s'])

    async def test_ambiguous_grader_error_is_not_retried(self):
        calls = []
        async def generate(payload):
            return result(7)
        async def verify(idx, answer):
            calls.append(idx)
            raise TimeoutError()
        summary = await run([{'problem_idx': 1, 'problem': 'Synthetic'}], generate, verify,
                            Config(attempts=3), lambda e: None, time.monotonic())
        self.assertEqual(calls, [1])
        self.assertEqual(summary['outcome'], 'grader_error')


if __name__ == '__main__':
    unittest.main()
