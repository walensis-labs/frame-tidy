# Frame Tidy

Frame Tidy is a free, open-source app for Aura digital photo frames. It finds
the people in the photos on your frames, so you can pick one person and remove
their photos from your frames in one go. It runs on your own computer, and you
approve every change before it is sent to Aura.

This project is not made by Aura and has no connection to it.

## Who it is for

You have one or more Aura frames with a lot of photos on them, and you want
someone gone from them: an ex, a photo you no longer want on display. Doing
that by hand in the Aura app means finding each photo yourself.

You should be comfortable installing a program from a terminal for now (see
"Install and first run"). There is no packaged download yet.

## What you need

- A computer. It has been built and run on a Mac. Windows and Linux are
  untested. Frame Tidy is not a phone app. A phone
  can be used once the computer has set things up (see "Using it from a
  phone").
- Your Aura account email and password.
- To run from source only: [uv](https://docs.astral.sh/uv/), a tool that installs and runs the app.
- Disk space. The face model is about 290 MB. Loading keeps a medium-size
  copy of each photo on the computer: one measured 12-megapixel photo came
  back at about 700 KB, so 3,000 photos is roughly 2 GB.
- Time. Loading copies about 0.09 seconds per photo when 12 download at once,
  so about 5 minutes for 3,000 photos. Reading the faces runs at about 50
  photos a second on an Apple-silicon Mac, and slower on older machines. No
  one has timed a complete real load from start to finish yet, so treat these
  as rough.

## Install and first run

**On a Mac:** download `Frame Tidy.dmg` from the
[Releases page](https://github.com/walensis-labs/frame-tidy/releases), open
it, and drag Frame Tidy into Applications. The app is signed and notarized by
Apple. It needs an Apple-silicon Mac (M1 or later) with macOS 14 or later. It checks for a newer version at most once a
day and shows a note when there is one; to update, download the new dmg the same
way. Your data stays where it is.

**From source** (any system, needs [uv](https://docs.astral.sh/uv/)):

```sh
git clone https://github.com/walensis-labs/frame-tidy.git
cd frame-tidy
uv sync
uv run aura-app
```

A Windows build is made by CI but has not been tested by anyone; see
`packaging/README.md`.

The app opens in its own window. If it cannot open a window it uses your
default browser. It then walks you through these steps:

1. **First, the face model.** The app tells you what it is (InsightFace's
   "buffalo_l" model, about 290 MB, downloaded once from InsightFace), links
   to InsightFace's terms, and waits. Nothing is downloaded until you press
   **I agree, download it**. Then it shows progress ("Downloading the face
   model...").
2. **Sign in.** A short notice appears above the form. Tick **I understand**
   (the Sign in button stays off until you do), then enter your Aura email
   and password and press **Sign in**.
3. **Which frames should I look through?** Tick the frames to use and press
   **Get my photos**. The photos are copied to this computer so faces can be
   found here. Nothing on your frames changes at this step.
4. **Getting your photos...** Leave the page open. People found so far appear
   once there are enough photos to group, and the page keeps updating as more
   arrive. Removing is switched off until loading has finished.
5. **Pick a person.** The **People** tab shows each person seen in two or more
   photos, with a face, a count, and a name if you gave one. People in ten
   or more photos are shown first. Smaller groups are folded under "Smaller
   groups", and people seen once under "Seen once". If no people are listed
   yet, press **Find people**.
   - **Rename** gives a person a name. **Hide** tucks them away. **Merge**
     on two tiles joins two groups that are the same person.
   - Click a person to see their photos. Every match arrives already ticked
     for removal. Press the cross on any photo that is not them: it is then
     unticked and left out.
   - With two or more frames loaded, the **All frames** view shows every
     photo once, with an "on 2 frames" badge on photos that are on several.
6. **Remove.** Press **Remove** and read the dialog (see the next sections).

Other tabs: **All** (every photo), **Flagged** (matches from a search, with
yes/no buttons to teach it), **Found** (search by description, such as "a dog"
or "a receipt"), **Selected** (what you have ticked) and **Removed** (what you
removed with the app). The **Frames** button brings back the frame list so you
can add a frame or pick up new photos.

## What stays on your computer

- Your photos. Medium-size copies (1,600 pixels on the long side) are kept in
  the app's data folder, and face reading happens on your computer. Nothing is
  uploaded anywhere. There is no server run by this project.
- Your names for people, your ticks, and the copies of removed photos.
- Your sign-in. Your password goes to Aura to sign in and is not saved. Aura
  replies with a sign-in token, which the app keeps in a file only you can
  read so you stay signed in. **Sign out of Aura** deletes that file.

The data folder is `~/Library/Application Support/Frame Tidy` on macOS,
`%APPDATA%\Frame Tidy` on Windows and `~/.local/share/Frame Tidy` on Linux.

What the app does contact on the internet:

- **Aura**, to sign in, list your frames, download your photos and remove
  the ones you approve.
- **InsightFace**, once, to download the face model, after you agree.
- **Hugging Face**, once, for the search model. The first time you press
  **Find** (search by description), the app explains the model (CLIP, in a
  version prepared by Qdrant, about 610 MB, MIT licence) and downloads it only
  after you agree. It is kept in `~/.frame-tidy/search-model`. Finding people
  does not need it.
- **GitHub**, at most once a day, to ask for the newest release number, so it
  can show an "update available" banner. It sends nothing about you. Set the
  environment variable `AURA_NO_UPDATE_CHECK` to anything to turn this off.

## Removing and putting back

Pressing **Remove** does not change anything yet. A dialog says exactly what
will happen: how many photos, and from which frames. If you have several
frames you choose between "Only from this frame" and "From every frame they
are on". It warns you if photos would end up on no frame at all, and about
photos you marked "not them" or picked by hand. Nothing is sent to Aura until
you press the red button, and only the photos the dialog listed are removed.

Be clear about what removal means:

- **It is permanent on Aura.** Aura has no trash. A photo that is no longer on
  any frame is gone from Aura.
- **Your safety copy.** Before Aura is asked to remove a photo, the app
  downloads the full-size original into a `removed` folder in the data folder
  and records it. It checks that the file opens as a whole photo and is at
  least as large as the copy already on the computer. If that fails, the photo
  is not removed, and you are told. Closing the app waits for the photo in
  progress to finish.
- **Photos that left the frame on their own.** Loading never deletes a local
  photo that is no longer on the frame. It moves it to `removed`, and the
  **Removed** tab lists it as gone missing from the frame, not removed by this
  app.
- **Put back.** The **Removed** tab lists the photos you removed with the app,
  newest first. Select some and press **Put back**. This works only while
  Aura still has the photo, which means it is still on another frame. If Aura
  no longer has it, the app tells you where the kept copy is, and you have to
  add that copy again with the Aura app.

## Using it from a phone

This is optional and off until you turn it on. Press **Use on my phone** on
the computer. The app opens the page on your wifi address and shows a QR code.
Scan it with the phone's camera. The link works once and for 10 minutes, and access is tied to the device that
scanned it. Press
Stop in the same dialog when you are done. It also stops when you quit the app.

- The phone and the computer must be on the same wifi.
- The connection is plain HTTP, not encrypted. Anyone on the same wifi could
  in principle watch what the phone sees, photos included. Use it only on a
  home network you trust.
- From the phone you can look through people and photos, mark "not them", and
  remove photos with the same confirmation. You cannot sign in, sign out or get
  photos from Aura on the phone, because a password must not cross the wifi
  unencrypted. The first set-up always happens on the computer. A paired phone is not shown
  file locations on the computer, only home-network address ranges are used,
  and the search-model download cannot be started from a phone.

## The honest limits

- **It is unofficial.** Frame Tidy talks to Aura's undocumented, private
  interface (the one Aura's own phone app uses, and it identifies itself to
  Aura as that phone app). Aura's terms of service
  (updated May 2021) do not allow automated tools like this. Whether to use
  it is your choice, and your account is the one at stake. Aura could block
  it or close an account.
- **It can stop working at any time.** Aura can change or close that interface
  without notice. When Aura refuses a request, the app says so in plain words
  (sign-in expired, change refused, Aura could not be reached, or an
  unexpected reply that may mean the app needs updating). It does not try to
  get around that.
- **It needs a computer.** There is no phone-only way to use it.
- **Face matching makes mistakes.** Relatives who look alike get mixed up.
  Small, blurry or side-on faces get missed. That is why you review and
  approve every removal. Check the photos before you press the red button.
- **The face model has its own licence.** InsightFace states that its
  pretrained models are available for non-commercial research purposes only.
  The code in this repository is MIT-licensed, but the model is not covered
  by that, and you agree to InsightFace's terms yourself in the first step.
  Read them if you are not sure that your use is covered.
- **Early software.** It has been tried on a real account and on a test
  service, not on many people's frames. The macOS app is unsigned and the
  Windows build is untested.

---

# For developers

Everything below is for people who build, test or extend the project, or who
want to use it from Claude through MCP.

The same code serves four commands, all defined in `pyproject.toml`:

| Command | What it is |
|---|---|
| `uv run aura-app` | The desktop app (own window, free port, per-user data folder). |
| `uv run aura-review` | The same page without the window, on `http://127.0.0.1:8766`. Data stays in the repository folder unless `AURA_MCP_HOME` says otherwise. |
| `uv run aura-mcp` | An MCP server for Claude (stdio). |
| `uv run aura-mock` | A stand-in for the Aura API, for testing. |

Two things differ between the app and the MCP server. The app asks before it
downloads the face model and the search model; the MCP tools download them on
first use, without asking (their descriptions say so). And the MCP
tools use `AURA_EMAIL` / `AURA_PASSWORD` from `.env`, or the sign-in the page
saved in `session.json`.

## Use from Claude

Put the login in `.env` (`cp .env.example .env`, then fill in `AURA_EMAIL` and
`AURA_PASSWORD`) and register the MCP server with Claude Code:

```sh
claude mcp add-json aura-frames '{"type":"stdio","command":"uv","args":["run","--directory","/path/to/frame-tidy","aura-mcp"]}'
```

or the equivalent `.mcp.json` entry:

```json
{
  "mcpServers": {
    "aura-frames": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/frame-tidy", "aura-mcp"]
    }
  }
}
```

Credentials are read from `.env` in the data folder (the repository folder by
default), so they do not need to go in the MCP config.

## Tools

| Tool | What it does |
|---|---|
| `list_frames()` | Lists your frames (`id`, `name`, `asset_count`). |
| `sync_frame_photos(frame_id)` | Downloads every photo on the frame to `cache/{frame_id}/` as medium-size (1600px) copies, 12 at a time, and writes `index.json`. Skips photos already cached. |
| `find_person_in_frame(frame_id, reference_image_path="", similarity_threshold=0.45, person="")` | Finds photos of a person, given either a reference photo or the name of someone saved in the review page (which also uses your yes/no answers). Copies matches to `flagged/{frame_id}/` for you to look through. Removes nothing. |
| `find_people()` | Groups the faces in every synced frame's cached photos into people. Scans photos not scanned before (about 0.2 s each), then takes seconds. People already found keep their id, name and hidden flag. Removes nothing. |
| `list_people()` | The people `find_people` found, most photographed first: those seen in two or more photos, the one-offs, and the hidden. Pass an id or name as `person` to `find_person_in_frame`. |
| `find_photos_in_frame(frame_id, query, similarity_threshold=0.22, top_k=100)` | Open-vocabulary search with a local CLIP model: give a phrase like `"a desk"`, `"a birthday cake"`, `"a receipt"` or `"a dog"` and it returns the best-matching photos, ranked. For curating a frame, not just cleaning it of a person. Writes `flagged/{frame_id}/description.json`; removes nothing. |
| `get_selection(frame_id)` | Returns the photos ticked in the review page, plus the saved people and reference faces. |
| `delete_assets(frame_id, asset_ids, scope="frame", confirm=False)` | With `confirm=False`, returns what *would* be removed. With `confirm=True`, removes the photos (see scopes below). A full-size copy of each one is downloaded into `removed/{frame_id}/` and recorded first; it is checked, and if that fails, that photo is not removed. |
| `restore_assets(frame_id, asset_ids)` | Puts removed photos back on the frame by re-linking the existing Aura photo. Works while Aura still has the photo, i.e. it is on another frame. |

Typical session: "list my Aura frames" → "sync the Kitchen frame" → "find
photos of the person in ~/Desktop/reference.jpg" → open `flagged/<frame_id>/`
and check the matches → "remove those" (or name the ones to keep).

### Delete scopes

| `scope` | Aura call | Effect |
|---|---|---|
| `"frame"` (default) | `POST /frames/{frame_id}/remove_asset.json` | Takes the photo off this frame only. It stays on any other frame it is on. A photo that is on no other frame is gone for good. |
| `"all_frames"` | `POST /frames/{frame_id}/remove_asset.json`, once per frame | Takes the photo off every frame it is on, among the frames synced here. The preview lists those frames, and names any frame that was never synced and so was not checked. |
| `"account"` | `DELETE /assets/{asset_id}.json` | Permanently deletes the photo from the account, removing it from every frame. Not offered in the review page or the app: it has not been run against the real service yet. |

Aura has no trash. A frame-scope removal can be reversed with
`restore_assets` as long as the photo is still on another frame; it comes
back as added by you, with today's date. Removing a photo from every frame,
from the only frame it is on, or from the account cannot be reversed in Aura:
the full original kept in `removed/` is then the only one, and has to be added
back through the Aura app.

Verified on a real account (2026-11-17): a contributor who does not own the
frame can remove, and re-link, photos uploaded by other people.

## Review page

```sh
uv run aura-review        # then open http://127.0.0.1:8766
```

The page the app wraps. It listens on `127.0.0.1` only (the phone listener,
when turned on, is a second one and is described above). Beyond what the
user section covers:

- **Select photos:** click a photo to tick it; shift-click selects a range.
  Selections are saved to `selections/{frame_id}.json` (and `selections/all.json`
  for the All frames view) as you go. `get_selection(frame_id)` returns them,
  so "remove the photos I selected" works from Claude Code, still through the
  `delete_assets` preview and confirmation.
- **Pick the person by hand:** open a photo with the corner button, click a
  face, name them and save. The face crop goes to `references/` and can be
  used as `reference_image_path` for `find_person_in_frame`.
- **Flagged tab and teaching it:** after a search, matches show with their
  similarity scores. Press the tick (it is them) or cross (not them) on each,
  or use "The other N are ..." to confirm the rest, then **Search again**.
  Confirmed faces are averaged into the person's reference. Rejected faces
  become "not this person" examples, which separates look-alike relatives.
  Answers are kept in `references/feedback.json` and apply to every later search
  for that name.
- **People:** found by clustering cached face embeddings (average linkage,
  cosine >= 0.40), pooled across frames by photo id. The store is
  `people/people.json`; face crops are in `people/avatars/`. "More of them?"
  offers other groups that look like the same person ("Same person" merges,
  "No" is remembered); **Undo last merge** reverses the latest merge until the
  next "Find people again".
- A photo marked "not them" is left out on every frame it is on. Opening a
  different person clears the previous person's ticks (hand-picked photos
  stay), so remove one person at a time.
- The `delete_assets` preview warns for each selected photo that was marked
  "not them", matched but left unanswered, or picked by hand. For a person
  opened from the People tab, unanswered matches are taken to be them and draw
  no warning; the preview reports them as `assumed_count`.
- The removal route uses a one-time preview token: only what the preview
  listed is removed.

## Things worth knowing

- **Review the flagged folder before confirming.** Face matching makes
  mistakes in both directions. Relatives who look alike are the usual false
  positive; side-on, small, or blurry faces are the usual misses.
- **Threshold.** `0.45` is a reasonable starting point. Lower it (e.g. `0.35`)
  to catch more photos at the cost of more false positives; raise it (e.g.
  `0.55`) for the reverse. Re-running with a different threshold or reference
  photo is fast because face embeddings are cached in
  `cache/{frame_id}/embeddings.npz`.
- **Reference photo.** Use a clear, front-facing photo containing only the
  target person. If it contains several faces, the largest is used and the
  result carries a warning.
- **Only synced photos can be removed.** `delete_assets` ignores (and reports)
  any ID that isn't in the frame's `index.json`.
- **Photos added by other people** may be refused by Aura when you try to
  delete them; those show up under `failed` in the report.
- **First use** of a face tool through MCP downloads the face model (~290 MB)
  to `~/.insightface` (or `AURA_MODEL_DIR`). Syncing is about 0.09 s per new
  photo with 12 downloads at once. Face reading is about 50 photos a second on
  an Apple-silicon Mac in the app's measurements. Both can be re-run to resume if
  interrupted.
- **Open-vocabulary search (`find_photos_in_frame`, and the "Found" tab in the
  review page)** uses a local CLIP model (ONNX via `fastembed`, ~610 MB, MIT-licensed, kept in
  `~/.frame-tidy/search-model` or `AURA_SEARCH_MODEL_DIR`; the app asks before
  downloading it, the MCP tool downloads on first use). CLIP similarities sit in a narrow band (~0.15–0.32),
  so the threshold defaults to `0.22` and results are **ranked** — read from the
  top and the best matches are first. A fixed cut-off is query-dependent
  (narrow queries like "a receipt" stay tight; broad ones like "a baby" match a
  lot), so lean on the ranking, not an absolute score. Image embeddings are
  cached in `cache/{frame_id}/clip_embeddings.npz`, so re-querying is instant.
- **People** are found by clustering the face embeddings the scans already
  cached (average linkage, cosine ≥ 0.40), pooled across frames by photo id,
  so a photo on several frames counts once. On the LFW test set each person's
  photos fell into one group and no group mixed two people. The store is
  `people/people.json`; face crops are in `people/avatars/`. One build handles
  up to 10,000 faces.
- `cache/`, `flagged/`, `removed/`, `selections/`, `references/`, `people/`, `mock/`, `session.json` and `.env` are gitignored. They hold personal photos and a login: never commit them.

## Face model

`buffalo_l` from InsightFace, running on CPU through ONNX Runtime:

- **Detection:** SCRFD-10GF (`det_10g.onnx`), finds every face in a photo.
- **Recognition:** ArcFace with a ResNet-50 backbone trained on WebFace600K
  (`w600k_r50.onnx`), turning each face into a 512-number embedding.

Two faces are compared by cosine similarity of their embeddings. The pack's
landmark and age/gender models are not loaded. InsightFace's pretrained
models are licensed for non-commercial research use only. The model lives in
`~/.insightface` unless `AURA_MODEL_DIR` says otherwise.

## Testing with a mock frame

`aura-mock` is a local stand-in for the Aura API, so the whole workflow can be
exercised without a real account. Each subfolder of `mock/photos/` is a frame;
each image in it is a photo on that frame. The same image file in two folders
is one photo on two frames. Removals only change the mock's memory: files on
disk are never touched, and a restart (or `POST /mock/reset`) restores
everything.

```sh
# 1. Test photos: the public LFW dataset (about 180 MB), then two mock frames
mkdir -p mock/downloads/lfw
curl -L -o mock/downloads/lfw.tgz https://ndownloader.figshare.com/files/5976018
tar -xzf mock/downloads/lfw.tgz -C mock/downloads/lfw
uv run python scripts/build_mock_frames.py        # --person Tony_Blair for someone else

# 2. One command: start the mock, run every tool over stdio, score the matching
uv run python scripts/mock_e2e.py                 # --threshold 0.35 to experiment
```

Or put your own photos in `mock/photos/<Frame Name>/` and drive it by hand
from Claude Code. Start the mock, then register a second server that points
at it and keeps its cache in `mock/`, apart from the real one:

```sh
uv run aura-mock
claude mcp add-json aura-frames-mock '{"type":"stdio","command":"uv","args":["run","--directory","/path/to/frame-tidy","aura-mcp"],"env":{"AURA_MCP_HOME":"/path/to/frame-tidy/mock","AURA_API_BASE_URL":"http://127.0.0.1:8765/v5","AURA_IMAGE_BASE_URL":"http://127.0.0.1:8765/img","AURA_EMAIL":"mock@example.com","AURA_PASSWORD":"mock"}}'
```

The mock accepts any email and password, except the password `wrong`, which
reproduces Aura's rejected-login response. `POST /mock/expire-tokens`
invalidates the session to exercise re-login; `GET /mock/state` shows the
frames and photo counts.

The mock copies the request and response shapes seen from the real API, but it
is a model of an undocumented service. It proves the tool logic, not that
Aura behaves the same way.

## Development

```sh
uv run pytest
```

Unit tests use a mocked HTTP transport and a fake face model; the workflow
tests run the real client against the mock API in-process. None of them
contact Aura or load the real model.

| Variable | Purpose | Default |
|---|---|---|
| `AURA_EMAIL`, `AURA_PASSWORD` | Aura login | from `.env` |
| `AURA_MCP_HOME` | Where `.env`, `session.json`, `cache/`, `flagged/` and the rest live | the repository folder (`aura-app`: the per-user app folder) |
| `AURA_API_BASE_URL` | API address | `https://api.pushd.com/v5` |
| `AURA_IMAGE_BASE_URL` | Image download address | `https://imgproxy.pushd.com` |
| `AURA_MODEL_DIR` | Where InsightFace keeps its models | `~/.insightface` |
| `AURA_SEARCH_MODEL_DIR` | Where the search model is kept | `~/.frame-tidy/search-model` |
| `AURA_NO_UPDATE_CHECK` | Set to anything to never look for a newer release | unset |

To run the app against the mock without touching a real account, set
`AURA_MCP_HOME` to a folder under `mock/`, point `AURA_API_BASE_URL` and
`AURA_IMAGE_BASE_URL` at the mock, set `AURA_NO_UPDATE_CHECK=1`, and make sure
`AURA_EMAIL` and `AURA_PASSWORD` are not set.

## Packaging and publishing

Building, signing and releasing the macOS and Windows apps: see
`packaging/README.md`.

## Licence

MIT for this project's code (see `LICENSE`). The InsightFace face model is not
part of the repository and has its own non-commercial licence.
