"""Quality test for gentle-zero model (5 experts zeroed in layer 0)."""
import time


def main():
    from vllm import LLM, SamplingParams

    print('Loading gentle-zero model...', flush=True)
    t0 = time.time()
    llm = LLM(
        model='/models/gemma4-gentle-zero',
        quantization='modelopt',
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
    )
    load_time = time.time() - t0
    print(f'Loaded in {load_time:.1f}s', flush=True)

    prompts = [
        'What is the capital of France?',
        'What is 17 * 23?',
        'Explain quantum computing in simple terms.',
        'Write a Python function to find the nth Fibonacci number.',
        'What are the three laws of thermodynamics?',
        'Who wrote Pride and Prejudice?',
        'What is the speed of light in a vacuum?',
        'Explain the difference between supervised and unsupervised learning.',
        'Write a haiku about the ocean.',
        'What causes the seasons on Earth?',
        'If x^2 - 5x + 6 = 0, find x.',
        'What is photosynthesis? Explain in 2 sentences.',
        'Name the planets in our solar system from the sun.',
        'What is the difference between RAM and ROM?',
        'Write a one-sentence definition of DNA.',
        'If a train travels 120 km in 2 hours, what is its average speed?',
        'What is the Pythagorean theorem?',
        'Name three programming languages and their primary use cases.',
        'What is the boiling point of water at sea level?',
        'Explain gravity in simple terms.',
    ]

    params = SamplingParams(max_tokens=128, temperature=0)
    print(f'Running {len(prompts)} prompts...', flush=True)
    t1 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t1

    coherent = 0
    for i, (prompt, out) in enumerate(zip(prompts, outputs)):
        text = out.outputs[0].text.strip()
        words = text.split()
        is_coh = (
            len(words) >= 3 and
            sum(1 for c in text if c.isalpha()) > len(text) * 0.3 and
            not text.startswith('{') and
            not all(c in '{}[]<>_' for c in text if not c.isspace())
        )
        if is_coh:
            coherent += 1
        status = 'OK' if is_coh else 'FAIL'
        print(f'[{status}] Q: {prompt}', flush=True)
        print(f'      A: {text[:200]}', flush=True)
        print(flush=True)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print('=' * 60, flush=True)
    print(f'COHERENT: {coherent}/{len(prompts)} ({100*coherent/len(prompts):.0f}%)', flush=True)
    print(f'Time: {gen_time:.1f}s, Tokens: {total_tokens}, Throughput: {total_tokens/gen_time:.1f} tok/s', flush=True)
    print('=' * 60, flush=True)

    return coherent, total_tokens, gen_time, outputs, prompts


if __name__ == '__main__':
    main()
