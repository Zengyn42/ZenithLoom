#!/usr/bin/env python3
import asyncio
import argparse
import sys
import os
import json
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Disable stdout buffering
sys.stdout.reconfigure(line_buffering=True)

async def main():
    parser = argparse.ArgumentParser(description="Grok Playwright Stream Bridge")
    parser.add_argument("prompt", type=str, help="The prompt to send to Grok")
    parser.add_argument("--url", type=str, help="Grok URL to resume chat", default="https://grok.com")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds", default=180)
    parser.add_argument("--user-data-dir", type=str, help="Path to Chrome user data dir", default=None)
    args = parser.parse_args()

    if args.user_data_dir:
        user_data_dir = os.path.abspath(args.user_data_dir)
    else:
        user_data_dir = os.path.join(os.path.dirname(__file__), "bot-chrome-data")

    async with async_playwright() as p:
        browser_args = [
            "--headless=new",
            "--password-store=basic",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check"
        ]
        
        # Clear display environment variables to prevent WSLg from showing a window
        clean_env = os.environ.copy()
        clean_env.pop("DISPLAY", None)
        clean_env.pop("WAYLAND_DISPLAY", None)

        try:
            ctx = await p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                executable_path="/usr/bin/google-chrome",
                headless=True,
                args=browser_args,
                no_viewport=True,
                env=clean_env,
            )
        except Exception as e:
            print(json.dumps({"error": f"Failed to launch Chrome: {e}"}), flush=True)
            return

        page = await ctx.new_page()

        try:
            print(f"DEBUG: Navigating to {args.url}...", file=sys.stderr, flush=True)
            try:
                await page.goto(args.url, wait_until="commit", timeout=30000)
            except Exception as e:
                print(f"DEBUG: Page load timed out or failed (ignoring): {e}", file=sys.stderr, flush=True)
            
            print("DEBUG: Proceeding to find input field...", file=sys.stderr, flush=True)
            await page.wait_for_timeout(2000)

            # Find input field
            input_selector = 'div.tiptap.ProseMirror, .ProseMirror, textarea'
            print(f"DEBUG: Looking for input field ({input_selector})...", file=sys.stderr, flush=True)
            input_field = page.locator(input_selector).first
            
            print("DEBUG: Clicking and typing prompt...", file=sys.stderr, flush=True)
            await input_field.click()
            await page.keyboard.type(args.prompt, delay=50) # Type like a human
            await page.wait_for_timeout(500)
            
            print("DEBUG: Pressing Enter...", file=sys.stderr, flush=True)
            await page.keyboard.press("Enter")
            
            # Try to click send button as fallback
            try:
                send_button = page.locator('button[aria-label="Submit"]:visible, button[aria-label*="Send"]:visible')
                if await send_button.count() > 0:
                    print("DEBUG: Clicking submit/send button...", file=sys.stderr, flush=True)
                    await send_button.last.click(timeout=3000, force=True)
                else:
                    svg_btn = page.locator('button:has(svg):visible').last
                    if await svg_btn.is_visible():
                        print("DEBUG: Clicking svg button...", file=sys.stderr, flush=True)
                        await svg_btn.click(timeout=3000, force=True)
            except Exception as e:
                print(f"DEBUG: Failed to click send button: {e}", file=sys.stderr, flush=True)
            
            print("DEBUG: Waiting for response to start...", file=sys.stderr, flush=True)
            await page.wait_for_timeout(2000)

            last_text = ""
            stable_count = 0
            # Broader selectors for Grok's response bubbles, excluding the input editor
            bubble_selector = 'div[data-testid="message-bubble-response"], div[data-testid="message-row"]:not(:has(.ProseMirror)), .message-row:not(:has(.ProseMirror)), .prose:not(.ProseMirror)'
            
            print("DEBUG: Monitoring for new response bubbles...", file=sys.stderr, flush=True)
            for i in range(args.timeout):
                await page.wait_for_timeout(1000)
                bubbles = page.locator(bubble_selector)
                count = await bubbles.count()
                
                if i % 10 == 0:
                    print(f"DEBUG: i={i}, Bubble count: {count}", file=sys.stderr, flush=True)

                if count > 0:
                    # Filter for response-specific bubbles first
                    response_bubbles = page.locator('div[data-testid="message-bubble-response"]')
                    rb_count = await response_bubbles.count()
                    
                    if rb_count > 0:
                        target_bubble = response_bubbles.last
                    else:
                        target_bubble = bubbles.last
                        
                    current_text = await target_bubble.inner_text()
                    
                    if i % 5 == 0:
                        print(f"DEBUG: Target bubble text (len={len(current_text)}): {current_text[:30]!r}", file=sys.stderr, flush=True)
                        if len(current_text.strip()) == 0:
                            # If text is empty, peek at the HTML to see what's going on
                            html_peek = await target_bubble.inner_html()
                            print(f"DEBUG: Bubble HTML peek (first 100 chars): {html_peek[:100]!r}", file=sys.stderr, flush=True)

                    if current_text and current_text.strip() != last_text.strip():
                        # Skip if it's exactly the prompt (to avoid echoing user input)
                        if current_text.strip() == args.prompt.strip() and last_text == "":
                            print("DEBUG: Skipping user message bubble...", file=sys.stderr, flush=True)
                            continue
                        if current_text.startswith(last_text):
                            diff = current_text[len(last_text):]
                            if diff:
                                print(diff, end="", flush=True)
                        else:
                            # Re-print if structure changed
                            print("\n" + current_text, end="", flush=True)
                        
                        last_text = current_text
                        stable_count = 0
                    elif current_text.strip() == last_text.strip() and last_text != "":
                        stable_count += 1
                        
                        # 只有在文本稳定且“发送按钮”重新出现时，才确定回复结束
                        # 如果没有发送按钮，说明还在生成中，即使文本没变也要多等一会儿
                        submit_visible = await page.locator('button[aria-label="Submit"]:visible, button[aria-label*="Send"]:visible').count() > 0
                        
                        if (stable_count >= 15) or (stable_count >= 5 and submit_visible): 
                            media_data = await page.evaluate("""() => {
                            const bubbles = document.querySelectorAll('div.message-bubble, [data-testid="message-bubble"], .message-bubble, [class*="message-bubble"]');
                            if (!bubbles.length) return { imgs: [], videos: [] };
                            const last = bubbles[bubbles.length - 1];
                            const imgs = Array.from(last.querySelectorAll('img')).map(i => i.src);
                            const videos = Array.from(last.querySelectorAll('source')).map(v => v.src).concat(Array.from(last.querySelectorAll('video')).map(v => v.src));
                            return { imgs, videos };
                        }""")
                            imgs = media_data.get('imgs', [])
                            videos = list(set(media_data.get('videos', [])))
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

            final_url = page.url
            print(json.dumps({"__session_url__": final_url}), file=sys.stderr, flush=True)

        except Exception as e:
            print(json.dumps({"error": f"Execution error: {e}"}), file=sys.stderr, flush=True)
        finally:
            await page.close()
            await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
