#!/usr/bin/env python3
import asyncio
import argparse
import sys
import json
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Disable stdout buffering
sys.stdout.reconfigure(line_buffering=True)

async def main():
    parser = argparse.ArgumentParser(description="Grok Playwright Stream Bridge")
    parser.add_argument("prompt", type=str, help="The prompt to send to Grok")
    parser.add_argument("--url", type=str, help="Grok project URL to resume chat", default="https://grok.com/project/50afbd22-0176-4dc8-a46e-43255b9943f7?tab=conversations")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds", default=180)
    parser.add_argument("--user-data-dir", type=str, help="Path to Chrome user data dir", default=None)
    args = parser.parse_args()

    if args.user_data_dir:
        user_data_dir = os.path.abspath(args.user_data_dir)
    else:
        user_data_dir = os.path.join(os.path.dirname(__file__), "bot-chrome-data")

    async with async_playwright() as p:
        browser_args = [
            "--disable-background-networking",
            "--disable-sync",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check"
        ]

        try:
            ctx = await p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                executable_path="/usr/bin/google-chrome",
                headless=True,
                args=browser_args,
                no_viewport=True,
            )
        except Exception as e:
            print(json.dumps({"error": f"Failed to launch Chrome: {e}"}), flush=True)
            return

        page = await ctx.new_page()

        try:
            await page.goto(args.url, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            # Find textarea and submit
            textarea = page.locator('textarea').first
            await textarea.fill(args.prompt, force=True)
            await page.wait_for_timeout(200)
            await page.keyboard.press("Enter")
            
            # Wait for generating to start and for the AI bubble to appear
            await page.wait_for_timeout(1000)

            last_text = ""
            stable_count = 0
            timeout_ms = args.timeout * 1000
            start_time = asyncio.get_event_loop().time() * 1000

            while (asyncio.get_event_loop().time() * 1000) - start_time < timeout_ms:
                await page.wait_for_timeout(500)  # 0.5s polling
                
                data = await page.evaluate("""() => {
                    const bubbles = document.querySelectorAll('div.message-bubble, [data-testid="message-bubble"], .message-bubble, [class*="message-bubble"]');
                    if (!bubbles || bubbles.length === 0) return { error: "0 bubbles" };
                    if (bubbles.length < 2) return { error: "1 bubble: " + bubbles[0].innerText.substring(0, 20) };
                    const last = bubbles[bubbles.length - 1];
                    return { text: last.innerText.trim(), count: bubbles.length };
                }""")

                if data:
                    if 'error' in data:
                        # Print debug info to stderr every 5 ticks
                        if stable_count % 5 == 0:
                            print(json.dumps({"debug": data['error']}), file=sys.stderr, flush=True)
                        stable_count += 1
                    elif 'text' in data:
                        current_text = data['text']
                        
                        # Ensure we are not just seeing the user's prompt echoed
                        if current_text and current_text != args.prompt.strip():
                            if current_text.startswith(last_text):
                                diff = current_text[len(last_text):]
                                if diff:
                                    sys.stdout.write(diff)
                                    sys.stdout.flush()
                                    last_text = current_text
                                    stable_count = 0
                                else:
                                    stable_count += 1
                            else:
                                # Text was completely replaced or restructured
                                sys.stdout.write("\\n" + current_text)
                                sys.stdout.flush()
                                last_text = current_text
                                stable_count = 0
                                
                            # If stable for 2.5s (5 ticks of 0.5s), assume completion
                            if stable_count >= 5:
                                media_data = await page.evaluate("""() => {
                                    const bubbles = document.querySelectorAll('div.message-bubble, [data-testid="message-bubble"], .message-bubble, [class*="message-bubble"]');
                                    const last = bubbles[bubbles.length - 1];
                                    const imgs = Array.from(last.querySelectorAll('img')).map(i => i.src);
                                    const videos = Array.from(last.querySelectorAll('source')).map(v => v.src).concat(Array.from(last.querySelectorAll('video')).map(v => v.src));
                                    return { imgs, videos };
                                }""")
                                if media_data:
                                    imgs = media_data.get('imgs', [])
                                    videos = list(set(media_data.get('videos', []))) # Deduplicate
                                    if imgs or videos:
                                        sys.stdout.write("\n\n")
                                        for img in imgs:
                                            if not img.startswith('data:image'):
                                                sys.stdout.write(f"![Image]({img})\n")
                                        for vid in videos:
                                            if vid:
                                                sys.stdout.write(f"[Video]({vid})\n")
                                        sys.stdout.flush()
                                break
                        else:
                            stable_count += 1

            final_url = page.url
            print(json.dumps({"__session_url__": final_url}), file=sys.stderr, flush=True)

        except Exception as e:
            print(json.dumps({"error": f"Execution error: {e}"}), file=sys.stderr, flush=True)
        finally:
            await page.close()
            await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
