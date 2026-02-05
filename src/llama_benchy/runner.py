import asyncio
import subprocess
import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Tuple
import aiohttp

from .config import BenchmarkConfig
from .client import LLMClient, RequestResult
from .prompts import PromptGenerator

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, client: LLMClient, prompt_generator: PromptGenerator):
        self.config = config
        self.client = client
        self.prompt_gen = prompt_generator
        self.results_table = []
        
        # We need to track deltas from warmup to adapt prompts
        self.delta_user = 0
        self.delta_context = 0

    async def run_suite(self):
        # Initialize session
        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=self.config.concurrency + 5, force_close=False, keepalive_timeout=600)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
            # Warmup
            should_warmup = not self.config.no_warmup
            if self.config.adapt_prompt:
                should_warmup = True
            
            if should_warmup:
                tokenizer = self.prompt_gen.corpus.get_tokenizer() if self.config.adapt_prompt else None
                self.delta_user, self.delta_context = await self.client.warmup(session, tokenizer)

            # Measure latency
            latency = await self.client.measure_latency(session, self.config.latency_mode)

            # Main Loop
            for depth in self.config.depths:
                for pp in self.config.pp_counts:
                    for tg in self.config.tg_counts:
                        print(f"Running test: pp={pp}, tg={tg}, depth={depth}, concurrency={self.config.concurrency}")
                        
                        # Storage for aggregated results across all runs
                        agg_results = {
                            "pp_speeds": [], "tg_speeds": [], "ttft_values": [], 
                            "ttfr_values": [], "est_ppt_values": [], "e2e_ttft_values": [],
                            "ctx_pp_speeds": [], "ctx_tg_speeds": [], "ctx_ttfr_values": [], 
                            "ctx_est_ppt_values": [], "ctx_e2e_ttft_values": [],
                            # Batch-level throughput tracking
                            "batch_pp_throughputs": [], 
                            "batch_tg_throughputs": [],
                            "ctx_batch_pp_throughputs": [], 
                            "ctx_batch_tg_throughputs": []
                        }

                        for run in range(self.config.num_runs):
                            
                            # Adapt prompt tokens
                            current_pp = pp
                            current_depth = depth
                            if self.config.adapt_prompt:
                                if depth == 0:
                                    current_pp = max(1, pp - self.delta_user)
                                else:
                                    current_depth = max(1, depth - self.delta_context)

                            prompt_batch = self.prompt_gen.generate_batch(
                                self.config.concurrency, 
                                current_pp, 
                                current_depth, 
                                self.config.no_cache
                            )
                            
                            if self.config.enable_prefix_caching and depth > 0:
                                # Phase 1: Context Load
                                print(f"  Run {run+1}/{self.config.num_runs} (Context Load, batch size {self.config.concurrency})...")
                                load_tasks = []
                                for i in range(self.config.concurrency):
                                    context, _ = prompt_batch[i]
                                    load_tasks.append(self.client.run_generation(
                                        session, 
                                        context_text=context, 
                                        prompt_text="", 
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache
                                    ))
                                
                                load_results = await asyncio.gather(*load_tasks)
                                self._process_batch_results(load_results, agg_results, current_depth, latency, is_context_phase=True)

                                # Phase 2: Inference
                                print(f"  Run {run+1}/{self.config.num_runs} (Inference, batch size {self.config.concurrency})...")
                                inf_tasks = []
                                for i in range(self.config.concurrency):
                                    context, prompt = prompt_batch[i]
                                    inf_tasks.append(self.client.run_generation(
                                        session,
                                        context_text=context,
                                        prompt_text=prompt,
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache
                                    ))
                                
                                batch_results = await asyncio.gather(*inf_tasks)
                                self._process_batch_results(batch_results, agg_results, current_pp, latency, is_context_phase=False)

                            else:
                                # Standard Run
                                print(f"  Run {run+1}/{self.config.num_runs} (batch size {self.config.concurrency})...")
                                expected_tokens = current_pp + current_depth
                                batch_tasks = []
                                for i in range(self.config.concurrency):
                                    context, prompt = prompt_batch[i]
                                    batch_tasks.append(self.client.run_generation(
                                        session,
                                        context_text=context,
                                        prompt_text=prompt,
                                        max_tokens=tg,
                                        no_cache=self.config.no_cache
                                    ))
                                
                                batch_results = await asyncio.gather(*batch_tasks)
                                self._process_batch_results(batch_results, agg_results, expected_tokens, latency, is_context_phase=False)
                            
                            # Post Run Command
                            if self.config.post_run_cmd:
                                try:
                                    subprocess.run(self.config.post_run_cmd, shell=True, check=True)
                                except subprocess.CalledProcessError as e:
                                    print(f"Post-run command failed: {e}")

                        self._record_results(agg_results, self.config.model, pp, tg, depth)

        self._print_final_report()

    def _process_batch_results(self, results: List[RequestResult], agg_results: Dict, expected_pp_tokens: int, latency: float, is_context_phase: bool):
        valid_results = [r for r in results if r and not r.error]
        if not valid_results:
            return

        # 1. Calculate Per-Request Metrics (Latencies & Individual Speeds)
        batch_prompt_tokens = 0
        batch_gen_tokens = 0
        
        # For batch throughput calculation
        start_times = []
        end_times = []
        first_token_times = []

        prefix = "ctx_" if is_context_phase else ""

        for res in valid_results:
            start_times.append(res.start_ts)
            end_times.append(res.end_ts)
            
            # Use reported usage if available and reasonable, else expected
            prompt_tokens = expected_pp_tokens
            if res.prompt_tokens > 0:
                diff = abs(res.prompt_tokens - expected_pp_tokens)
                if diff < expected_pp_tokens * 0.2:
                    prompt_tokens = res.prompt_tokens
            
            batch_prompt_tokens += prompt_tokens
            batch_gen_tokens += res.total_tokens

            # Metrics Calculation
            ttft = 0.0
            e2e_ttft = 0.0
            ttfr = 0.0
            est_ppt = 0.0
            
            if res.first_response_ts:
                ttfr = res.first_response_ts - res.start_ts
                agg_results[f"{prefix}ttfr_values"].append(ttfr)
            
            if res.first_token_ts:
                first_token_times.append(res.first_token_ts)
                e2e_ttft = res.first_token_ts - res.start_ts
                ttft = max(0, e2e_ttft - latency)
                est_ppt = max(0, ttfr - latency) # Estimate PPT based on first response time - latency

                agg_results[f"{prefix}e2e_ttft_values"].append(e2e_ttft)
                agg_results[f"{prefix}ttft_values"].append(ttft)
                agg_results[f"{prefix}est_ppt_values"].append(est_ppt)

            # Individual Speeds
            # PP Speed
            if est_ppt > 0:
                pp_speed = prompt_tokens / est_ppt
                agg_results[f"{prefix}pp_speeds"].append(pp_speed)
            
            # TG Speed
            if res.total_tokens > 1 and res.first_token_ts:
                decode_time = res.end_ts - res.first_token_ts
                if decode_time > 0:
                    tg_speed = (res.total_tokens - 1) / decode_time
                    agg_results[f"{prefix}tg_speeds"].append(tg_speed)

        # 2. Calculate Batch-Level Throughput
        if start_times and end_times and first_token_times:
            min_start = min(start_times)
            max_end = max(end_times)
            
            # PP Throughput: Total Prompt Tokens / Time from First Start to Last First Token
            # Why? Because PP is considered "done" when the first token appears.
            # In a batch, the system is processing prompts from min_start until the last request starts generating.
            max_first_token = max(first_token_times)
            pp_duration = max_first_token - min_start
            
            if pp_duration > 0:
                batch_pp_throughput = batch_prompt_tokens / pp_duration
                agg_results[f"{prefix}batch_pp_throughputs"].append(batch_pp_throughput)
            
            # TG Throughput: Total Gen Tokens / Time from First Token (Start of Gen) to Last End
            # Why? Generation window is from the moment the first request starts generating tokens 
            # (indicating system has switched to decode phase for at least one req) until the last one finishes.
            # Note: This assumes some overlap. Strict throughput is (Count / WallTime).
            min_first_token = min(first_token_times)
            tg_duration = max_end - min_first_token
            
            # For accurate "Token Generation Only" throughput, we consider the generated tokens (minus the first one which is part of TTFT response usually)
            # But roughly Total Tokens is fine.
            if tg_duration > 0:
                batch_tg_throughput = (batch_gen_tokens - len(valid_results)) / tg_duration  # Exclude first tokens? Often TG speed excludes first token.
                # If we have mainly TG heavy load, we can just use total tokens, but removing the first token (which was generated during PP/TTFT phase technically) is cleaner for "Decode Throughput"
                
                # If for some reason batch_gen_tokens is small, fallback
                if batch_gen_tokens <= len(valid_results):
                     batch_tg_throughput = batch_gen_tokens / tg_duration
                
                agg_results[f"{prefix}batch_tg_throughputs"].append(batch_tg_throughput)


    def _record_results(self, agg_results: Dict[str, List[float]], model: str, pp: int, tg: int, depth: int):
        def format_result(values, multiplier=1.0):
            if not values: return ""
            mean = np.mean(values) * multiplier
            std = np.std(values) * multiplier
            return f"{mean:.2f} ± {std:.2f}"

        # Context PP (if enabled)
        if agg_results["ctx_pp_speeds"] or (self.config.concurrency > 1 and agg_results["ctx_batch_pp_throughputs"]):
            test_name = f"ctx_pp @ d{depth}"
            pp_metric = agg_results["ctx_batch_pp_throughputs"] if self.config.concurrency > 1 else agg_results["ctx_pp_speeds"]
            
            self.results_table.append([
                model, 
                test_name, 
                format_result(pp_metric), 
                format_result(agg_results["ctx_ttfr_values"], 1000), 
                format_result(agg_results["ctx_est_ppt_values"], 1000), 
                format_result(agg_results["ctx_e2e_ttft_values"], 1000)
            ])

        # Context TG (if enabled)
        if agg_results["ctx_tg_speeds"] or (self.config.concurrency > 1 and agg_results["ctx_batch_tg_throughputs"]):
            test_name = f"ctx_tg @ d{depth}"
            tg_metric = agg_results["ctx_batch_tg_throughputs"] if self.config.concurrency > 1 else agg_results["ctx_tg_speeds"]
            self.results_table.append([model, test_name, format_result(tg_metric), "", "", ""])

        # Standard PP
        if agg_results["pp_speeds"] or (self.config.concurrency > 1 and agg_results["batch_pp_throughputs"]):
            test_name = f"pp{pp}"
            if depth > 0: test_name += f" @ d{depth}"
            pp_metric = agg_results["batch_pp_throughputs"] if self.config.concurrency > 1 else agg_results["pp_speeds"]
            
            self.results_table.append([
                model, 
                test_name, 
                format_result(pp_metric), 
                format_result(agg_results["ttfr_values"], 1000), 
                format_result(agg_results["est_ppt_values"], 1000), 
                format_result(agg_results["e2e_ttft_values"], 1000)
            ])
        
        # Standard TG
        if agg_results["tg_speeds"] or (self.config.concurrency > 1 and agg_results["batch_tg_throughputs"]):
            test_name = f"tg{tg}"
            if depth > 0: test_name += f" @ d{depth}"
            tg_metric = agg_results["batch_tg_throughputs"] if self.config.concurrency > 1 else agg_results["tg_speeds"]
            self.results_table.append([model, test_name, format_result(tg_metric), "", "", ""])

    def _print_final_report(self):
        print()
        if not self.results_table:
            print("No results collected. Check if the model is generating tokens.")
        else:
            print(tabulate(self.results_table, headers=["model", "test", "t/s (total)" if self.config.concurrency > 1 else "t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right")))
