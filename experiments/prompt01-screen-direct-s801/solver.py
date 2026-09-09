"""Concurrent one-GPU solver and serial grader client. No gold access."""
import argparse
import asyncio
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time


@dataclass
class Config:
    model: str = 'aime-model'
    concurrency: int = 30
    max_tokens: int = 8192
    retry_tokens: int = 16384
    attempts: int = 1
    seed: int = 42
    deadline_s: float = 900
    stop_at: int = 30
    temperature: float = 1.0
    top_p: float = 0.95
    finalize_tokens: int = 0
    probe_policy: str = 'none'
    probe_tokens: int = 8192
    overlap_probe: int = 0
    prompt_style: str = 'original'
    samples: int = 1
    queue_policy: str = 'fifo'


def pop_candidate(ready, votes, policy):
    if policy == 'agreement':
        index = max(range(len(ready)), key=lambda i: votes.get((ready[i][0]['problem_idx'], ready[i][2]), 0))
        ready.rotate(-index)
        selected = ready.popleft()
        ready.rotate(index)
        return selected
    return ready.popleft()


def problem_prompt(problem, style):
    if style == 'original':
        instruction = 'Solve the problem. Give your final integer answer in \\boxed{}.'
    elif style == 'concise':
        instruction = ('Solve directly and concisely using one clear method. '
                       'Avoid repeating derivations or checking the same result more than once. '
                       'Once the answer is justified, finish immediately with your final integer answer in \\boxed{}.')
    elif style in {'direct': 'Solve using one clear method. Avoid repeating derivations or rechecking a result already established. Stop when the answer is justified. Give your final integer answer in \\boxed{}.', 'checkpoint': 'Solve the problem step by step. At the end of each major step, assess whether the requested integer is already determined. If it is, finish instead of exploring another approach. Give your final integer answer in \\boxed{}.', 'answer_first': 'Identify the exact integer the problem asks for, then choose the shortest sound route to determine it. Compute only what is needed for that result and avoid unrelated intermediate quantities. Give your final integer answer in \\boxed{}.'}:
        instruction = {'direct': 'Solve using one clear method. Avoid repeating derivations or rechecking a result already established. Stop when the answer is justified. Give your final integer answer in \\boxed{}.', 'checkpoint': 'Solve the problem step by step. At the end of each major step, assess whether the requested integer is already determined. If it is, finish instead of exploring another approach. Give your final integer answer in \\boxed{}.', 'answer_first': 'Identify the exact integer the problem asks for, then choose the shortest sound route to determine it. Compute only what is needed for that result and avoid unrelated intermediate quantities. Give your final integer answer in \\boxed{}.'}[style]
    else:
        raise ValueError('Unknown prompt style')
    return problem + '\n\n' + instruction


def extract_answer(content, finish_reason):
    if finish_reason != 'stop' or not isinstance(content, str):
        return None
    if '<think>' in content and '</think>' not in content:
        return None
    final = content.rsplit('</think>', 1)[-1]
    matches = re.findall(r'\\boxed\s*\{\s*(\d{1,3})\s*\}', final)
    return str(int(matches[-1])) if matches else None


def load_questions(path):
    questions = []
    seen = set()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {'problem_idx', 'problem'}:
                raise ValueError('Only question-only input is accepted')
            idx = row['problem_idx']
            if type(idx) is not int or idx in seen or not isinstance(row['problem'], str):
                raise ValueError('Invalid or duplicate question')
            seen.add(idx)
            questions.append(row)
    if not questions:
        raise ValueError('Empty question set')
    return questions


async def run(questions, generate, verify, config, emit, started):
    """Services are injectable to test scheduling without real model or gold."""
    clock = lambda: time.monotonic() - started
    order = list(questions)
    random.Random(config.seed).shuffle(order)
    waiting = deque((q, sample, None) for sample in range(config.samples) for q in order)
    ready = deque()
    active = {}
    grading = None
    solved = set()
    submitted = set()
    votes = Counter()
    milestones = {}
    counts = Counter()
    completed_tokens = 0
    first_candidate_s = None
    first_submission_s = None
    grader_busy_s = 0.0
    outcome = 'exhausted'
    probe_successes = probe_failures = 0
    emit({'event': 'start', 'elapsed_s': clock(), 'config': asdict(config),
          'n_problems': len(questions), 'order': [q['problem_idx'] for q in order]})

    def retry(q, attempt):
        if attempt + 1 < config.attempts and q['problem_idx'] not in solved:
            waiting.append((q, attempt + 1, None))

    async def infer(q, attempt, resume=None):
        requested = clock()
        budget = config.max_tokens if attempt == 0 or config.samples > 1 else config.retry_tokens
        total_budget = budget
        if config.probe_policy != 'none' and resume is None:
            budget = config.probe_tokens
        payload = {
            'model': config.model,
            'messages': [{'role': 'user', 'content': problem_prompt(q['problem'], config.prompt_style)}],
            'temperature': config.temperature, 'top_p': config.top_p,
            'top_k': -1, 'max_tokens': budget,
            'seed': config.seed + q['problem_idx'] * 1009 + attempt,
        }
        if resume is not None:
            budget = resume['remaining_tokens']
            payload.update({'max_tokens': budget,
                            'messages': payload['messages'] + [
                                {'role': 'assistant', 'content': resume['reasoning']}],
                            'continue_final_message': True, 'add_generation_prompt': False,
                            'seed': payload['seed'] + 1000000})
        response = await generate(payload)
        first_choice = response['choices'][0]
        if resume is not None:
            first_choice['message']['content'] = resume['reasoning'] + (first_choice['message'].get('content') or '')
        next_stage = None
        if (config.probe_policy != 'none' and resume is None
                and first_choice['finish_reason'] == 'length'):
            next_stage = {'reasoning': first_choice['message'].get('content') or '',
                          'remaining_tokens': total_budget - budget,
                          'estimated_continuation_s': (clock() - requested) * (total_budget - budget) / budget}
        if first_choice['finish_reason'] == 'length' and config.finalize_tokens:
            reasoning = first_choice['message'].get('content') or ''
            initial_usage = response.get('usage') or {}
            counts['truncated'] += 1
            counts['finalization_requests'] += 1
            counts['generation_requests'] += 1
            emit({'event': 'reasoning_truncated', 'elapsed_s': clock(),
                  'index': q['problem_idx'], 'attempt': attempt,
                  'usage': initial_usage, 'content': reasoning})
            # Preserve the model's reasoning, close the reasoning section, and
            # ask it to complete a final-answer prefix. No answer is supplied.
            prefix = reasoning + '\n</think>\n\nThe final answer is \\boxed{'
            final_payload = dict(payload)
            final_payload.update({
                'messages': payload['messages'][:1] + [{'role': 'assistant', 'content': prefix}],
                'continue_final_message': True, 'add_generation_prompt': False,
                'max_tokens': config.finalize_tokens, 'stop': ['}'],
                'seed': payload['seed'] + 500000,
            })
            response = await generate(final_payload)
            choice = response['choices'][0]
            continuation = choice['message'].get('content') or ''
            closing = '}' if choice.get('stop_reason') == '}' else ''
            choice['message']['content'] = prefix + continuation + closing
            final_usage = response.get('usage') or {}
            response['usage'] = {key: initial_usage.get(key, 0) + final_usage.get(key, 0)
                                 for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            emit({'event': 'finalization', 'elapsed_s': clock(), 'index': q['problem_idx'],
                  'attempt': attempt, 'content': continuation, 'usage': final_usage,
                  'finish_reason': choice['finish_reason'], 'stop_reason': choice.get('stop_reason')})
        response['_resume'] = next_stage
        return response, requested, clock(), budget

    try:
        while waiting or active or ready or grading:
            if clock() >= config.deadline_s:
                outcome = 'deadline'
                break
            while waiting and len(active) < config.concurrency:
                q, attempt, resume = waiting.popleft()
                if q['problem_idx'] in solved:
                    continue
                task = asyncio.create_task(infer(q, attempt, resume))
                if resume is not None and resume.get('speculative'):
                    counts['speculative_continuations_started'] += 1
                    emit({'event': 'speculation_started', 'elapsed_s': clock(), 'index': q['problem_idx']})
                active[task] = (q, attempt)
                counts['generation_requests'] += 1
            if grading is None and ready:
                q, attempt, answer, resume = pop_candidate(ready, votes, config.queue_policy)
                key = (q['problem_idx'], answer)
                if q['problem_idx'] in solved or key in submitted:
                    counts['duplicates_skipped'] += 1
                    if resume is not None:
                        waiting.append((q, attempt, resume))
                    else:
                        retry(q, attempt)
                    continue
                submitted.add(key)
                grade_start = clock()
                if first_submission_s is None:
                    first_submission_s = grade_start
                emit({'event': 'submit', 'elapsed_s': grade_start, 'index': q['problem_idx'],
                      'attempt': attempt, 'candidate': answer, 'local_queue_size': len(ready),
                      'votes_at_submission': votes[(q['problem_idx'], answer)]})
                grading = asyncio.create_task(verify(q['problem_idx'], answer))
                grade_item = q, attempt, answer, grade_start, resume
                counts['verification_requests'] += 1
            pending = set(active)
            if grading is not None:
                pending.add(grading)
            if not pending:
                continue
            done, _ = await asyncio.wait(pending, timeout=max(0, config.deadline_s - clock()),
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                outcome = 'deadline'
                break
            # Receipt of a verdict is the scored milestone, not server timestamps.
            if grading in done:
                received = clock()
                q, attempt, answer, grade_start, resume = grade_item
                try:
                    verdict = grading.result()
                    if type(verdict.get('verdict')) is not bool:
                        raise ValueError('Invalid grader verdict')
                except Exception as exc:
                    emit({'event': 'grader_error', 'elapsed_s': received,
                          'error_type': type(exc).__name__})
                    outcome = 'grader_error'
                    break  # Never retry an ambiguous verification automatically.
                grader_busy_s += received - grade_start
                correct = verdict['verdict']
                if resume is not None:
                    probe_successes += int(correct)
                    probe_failures += int(not correct)
                counts['verified_correct' if correct else 'verified_incorrect'] += 1
                if correct:
                    solved.add(q['problem_idx'])
                    milestones[str(len(solved))] = received
                    if (config.overlap_probe and resume is not None) or config.samples > 1:
                        for other, (other_q, _) in active.items():
                            if other_q['problem_idx'] == q['problem_idx'] and not other.done():
                                other.cancel()
                                counts['sample_cancellations_requested' if config.samples > 1
                                       else 'speculative_cancellations_requested'] += 1
                else:
                    if resume is not None:
                        counts['continuations_after_failed_probe'] += 1
                        if not config.overlap_probe:
                            waiting.append((q, attempt, resume))
                    else:
                        retry(q, attempt)
                emit({'event': 'verdict', 'elapsed_s': received, 'index': q['problem_idx'],
                      'attempt': attempt, 'candidate': answer, 'correct': correct,
                          'solved_count': len(solved), 'request_elapsed_s': received - grade_start,
                          'early_probe': resume is not None})
                grading = None
                if len(solved) >= config.stop_at:
                    outcome = 'target'
                    break
            for task in list(active):
                if task not in done:
                    continue
                q, attempt = active.pop(task)
                try:
                    response, requested, received, budget = task.result()
                    choice = response['choices'][0]
                    content = choice['message'].get('content') or ''
                    finish = choice['finish_reason']
                    answer = extract_answer(content, finish)
                    usage = response.get('usage') or {}
                    resume = response.get('_resume')
                    completed_tokens += usage.get('completion_tokens', 0)
                    counts['completed_generations'] += 1
                    if finish == 'length':
                        counts['truncated'] += 1
                    if answer is None:
                        counts['no_final_answer'] += 1
                    else:
                        votes[(q['problem_idx'], answer)] += 1
                    emit({'event': 'generation', 'elapsed_s': received, 'requested_s': requested,
                          'index': q['problem_idx'], 'attempt': attempt, 'max_tokens': budget,
                          'finish_reason': finish, 'candidate': answer, 'usage': usage,
                          'content': content})
                    if answer is not None:
                        if first_candidate_s is None:
                            first_candidate_s = received
                        buy_probe = True
                        if resume is not None:
                            # Online Beta(1,1) estimate from this run's settled early checks.
                            # This is a latency surrogate, not a calibrated optimal VOI.
                            p_success = (probe_successes + 1) / (probe_successes + probe_failures + 2)
                            current_remaining = max(0.0, 3.0 - (clock() - grade_item[3])) if grading else 0.0
                            wait_s = current_remaining + 3.0 * (len(ready) + 1)
                            saved_s = p_success * resume['estimated_continuation_s']
                            buy_probe = config.probe_policy == 'always' or saved_s > wait_s
                            counts['probes_bought' if buy_probe else 'probes_skipped'] += 1
                            emit({'event': 'probe_decision', 'elapsed_s': clock(),
                                  'index': q['problem_idx'], 'buy': buy_probe,
                                  'policy': config.probe_policy, 'p_success': p_success,
                                  'estimated_saved_s': saved_s, 'estimated_wait_s': wait_s,
                                  'continuation_s': resume['estimated_continuation_s']})
                        if buy_probe:
                            ready.append((q, attempt, answer, resume))
                            if config.overlap_probe and resume is not None:
                                waiting.append((q, attempt, dict(resume, speculative=True)))
                        else:
                            waiting.append((q, attempt, resume))
                    else:
                        if resume is not None:
                            counts['continuations_without_candidate'] += 1
                            waiting.append((q, attempt, resume))
                        else:
                            retry(q, attempt)
                except asyncio.CancelledError:
                    counts['cancelled_generation_requests'] += 1
                    emit({'event': 'generation_cancelled', 'elapsed_s': clock(),
                          'index': q['problem_idx'], 'reason': 'already_verified'})
                except Exception as exc:
                    counts['generation_errors'] += 1
                    emit({'event': 'generation_error', 'elapsed_s': clock(),
                          'index': q['problem_idx'], 'attempt': attempt,
                          'error_type': type(exc).__name__})
                    retry(q, attempt)
    finally:
        pending = list(active) + ([grading] if grading is not None else [])
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    summary = {'event': 'summary', 'outcome': outcome, 'elapsed_s': clock(),
               'n_problems': len(questions), 'solved': len(solved),
               'verified_fraction': len(solved) / len(questions),
               'time_to_18_s': milestones.get('18'), 'milestones_s': milestones,
               'counts': dict(counts), 'completed_response_tokens': completed_tokens,
               'first_candidate_s': first_candidate_s, 'first_submission_s': first_submission_s,
               'grader_busy_s': grader_busy_s, 'config': asdict(config)}
    emit(summary)
    return summary


async def main(args):
    import aiohttp
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    if config.concurrency < 1 or config.attempts < 1 or config.deadline_s <= 0:
        raise ValueError('Invalid run budget')
    if config.samples < 1 or (config.samples > 1 and
                             (config.attempts != 1 or config.probe_policy != 'none')):
        raise ValueError('Independent samples require no retries or early probes')
    if config.queue_policy not in ('fifo', 'agreement') or (config.queue_policy == 'agreement' and config.samples < 2):
        raise ValueError('Agreement scheduling requires multiple independent samples')
    if config.probe_policy not in ('none', 'always', 'priced'):
        raise ValueError('Unknown probe policy')
    if config.overlap_probe not in (0, 1) or (config.overlap_probe and config.probe_policy != 'always'):
        raise ValueError('Overlap experiment requires the always-probe policy')
    if config.probe_policy != 'none' and (not 0 < config.probe_tokens < config.max_tokens
                                         or config.finalize_tokens < 1 or config.attempts != 1):
        raise ValueError('Probe policy needs one attempt, finalization, and a smaller checkpoint budget')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=config.deadline_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def get(url):
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        model_info = await get(args.model_url + '/v1/models')
        health = await get(args.grader_url + '/health')
        if not health.get('ok') or health.get('cost_c', 0) < 3:
            raise ValueError('Grader unavailable or toll below challenge requirement')
        if health.get('queries_so_far') != 0:
            raise ValueError('Start a fresh grader process for each run')
        metadata = {'utc': datetime.now(timezone.utc).isoformat(), 'models': model_info,
                    'grader_health': health, 'config': asdict(config),
                    'solver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        (output / 'solver.py').write_bytes(Path(__file__).read_bytes())

        async def post(url, body):
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                return await response.json()
        async def generate(payload):
            return await post(args.model_url + '/v1/chat/completions', payload)
        async def verify(idx, answer):
            return await post(args.grader_url + '/verify',
                              {'index': idx, 'candidate': answer, 'agent_id': output.name})

        with (output / 'events.jsonl').open('w', encoding='utf-8') as trace:
            def emit(event):
                trace.write(json.dumps(event, ensure_ascii=False) + '\n')
                trace.flush()
                if event['event'] in ('verdict', 'summary', 'grader_error'):
                    print(json.dumps({k: v for k, v in event.items() if k != 'milestones_s'}), flush=True)
            started = time.monotonic()  # Before receiving any problem text.
            questions = load_questions(args.questions)
            summary = await run(questions, generate, verify, config, emit, started)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-url', default='http://127.0.0.1:8000')
    parser.add_argument('--grader-url', default='http://127.0.0.1:8077')
    for key, default in asdict(Config()).items():
        parser.add_argument('--' + key.replace('_', '-'), default=default, type=type(default))
    asyncio.run(main(parser.parse_args()))


if __name__ == '__main__':
    cli()
