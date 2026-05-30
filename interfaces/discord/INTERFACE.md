# Discord Interface — Framework Contract

This document is automatically injected by the framework at Discord startup and applies to all Agents running via Discord.

## Sending Files to Discord

Include the following tag anywhere in your reply text to have the framework automatically send a file to the current channel, **without needing to know the channel ID**:

```
[SEND_FILE: /absolute/path/to/file]
```

Rules:
- The path must be an absolute WSL path.
- A single reply can contain multiple `[SEND_FILE: ...]` tags.
- The tag is automatically stripped from the displayed text; the user only sees the file attachment.
- The framework sends a warning if the file does not exist.

Example:
```
Video generation complete! [SEND_FILE: /tmp/output/video.mp4]
```

## Discord Channel Tools

When you need to query channel information, route to the `discord_tools` node and specify the operation via the `context`:

| Operation      | `context` Format                                  | Description                                       |
|----------------|---------------------------------------------------|---------------------------------------------------|
| Read History   | `history` or `history limit=N`                    | Defaults to the last **20** messages, max 200.      |
| Search Messages| `search q=<keyword>` or `search q=<keyword> limit=N` | Searches the last 100 messages by default.        |
| List Channels  | `channels`                                        | Lists all text channels on the server.            |
| Query User     | `user id=<USER_ID>`                               | Queries username, roles, and join date.           |

Example:
```json
{"route": "discord_tools", "context": "history limit=20"}
{"route": "discord_tools", "context": "search q=Video generation"}
{"route": "discord_tools", "context": "channels"}
{"route": "discord_tools", "context": "user id=286003878997262337"}
```

## User Identity

The message format is `username (DISCORD): message content`. `username` is the current speaker.
