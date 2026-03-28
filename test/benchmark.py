"""
FastAPI 新闻分类服务压测脚本
使用 asyncio + aiohttp 进行高并发压测
"""

import asyncio
import aiohttp
import time
import random
import argparse
from collections import Counter

# ============================================================
# 测试数据
# ============================================================
TEST_QUERIES = [
    "国足在世界杯预选赛中击败对手晋级下一轮",
    "股市大盘今日上涨3%，券商板块领涨",
    "新款手机发布引发抢购热潮，首日销量破百万",
    "某明星宣布退出娱乐圈，粉丝震惊",
    "我国新型战斗机成功完成首飞测试",
    "NBA总决赛今日开打，湖人对阵凯尔特人",
    "央行宣布降准0.5个百分点释放流动性",
    "人工智能大模型技术取得重大突破",
    "热门电影首周末票房突破10亿",
    "海军新型驱逐舰正式入列服役",
    "中超联赛新赛季即将开幕",
    "比特币价格突破历史新高",
    "苹果公司发布最新操作系统更新",
    "知名导演新片定档春节上映",
    "空军举行大规模实战演练",
]


async def send_request(session, url, query, results):
    """发送单条请求并记录结果"""
    payload = {"query": query}
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            status = resp.status
            await resp.json()
            latency = (time.perf_counter() - start) * 1000  # ms
            results.append({
                "status": status,
                "latency": latency,
                "success": status == 200,
            })
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        results.append({
            "status": 0,
            "latency": latency,
            "success": False,
            "error": str(e),
        })


async def run_benchmark(url, total_requests, concurrency):
    """执行压测"""
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(session, query):
        async with semaphore:
            await send_request(session, url, query, results)

    print(f"{'=' * 60}")
    print(f"🚀 新闻分类服务压测")
    print(f"{'=' * 60}")
    print(f"  目标地址:   {url}")
    print(f"  总请求数:   {total_requests}")
    print(f"  并发数:     {concurrency}")
    print(f"{'=' * 60}\n")

    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=60)

    start_time = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i in range(total_requests):
            query = random.choice(TEST_QUERIES)
            tasks.append(limited_request(session, query))
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    # ============================================================
    # 统计结果
    # ============================================================
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successes]

    if latencies:
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        p50 = latencies[int(len(latencies) * 0.50)]
        p90 = latencies[int(len(latencies) * 0.90)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(min(len(latencies) * 0.99, len(latencies) - 1))]
        min_latency = latencies[0]
        max_latency = latencies[-1]
    else:
        avg_latency = p50 = p90 = p95 = p99 = min_latency = max_latency = 0

    qps = total_requests / total_time if total_time > 0 else 0

    print(f"{'=' * 60}")
    print(f"📊 压测结果")
    print(f"{'=' * 60}")
    print(f"  总耗时:       {total_time:.2f}s")
    print(f"  总请求数:     {total_requests}")
    print(f"  成功:         {len(successes)}")
    print(f"  失败:         {len(failures)}")
    print(f"  成功���:       {len(successes)/total_requests*100:.2f}%")
    print(f"  QPS:          {qps:.2f} req/s")
    print()
    print(f"  📈 延迟统计 (ms)")
    print(f"  {'─' * 40}")
    print(f"  平均:         {avg_latency:.1f}")
    print(f"  最小:         {min_latency:.1f}")
    print(f"  最大:         {max_latency:.1f}")
    print(f"  P50:          {p50:.1f}")
    print(f"  P90:          {p90:.1f}")
    print(f"  P95:          {p95:.1f}")
    print(f"  P99:          {p99:.1f}")
    print(f"{'=' * 60}")

    # 错误统计
    if failures:
        print(f"\n  ❌ 错误详情:")
        status_counts = Counter(r.get("status", 0) for r in failures)
        for status, count in status_counts.items():
            print(f"    状态码 {status}: {count} 次")

    return results


def main():
    parser = argparse.ArgumentParser(description="新闻分类服务压测工具")
    parser.add_argument("--url", default="http://localhost:8000/classify",
                        help="API 地址 (默认: http://localhost:8000/classify)")
    parser.add_argument("-n", "--requests", type=int, default=100,
                        help="总请求数 (默认: 100)")
    parser.add_argument("-c", "--concurrency", type=int, default=10,
                        help="并发数 (默认: 10)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()