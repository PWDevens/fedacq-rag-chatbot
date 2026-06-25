# FedAcq RAG Chatbot — Plain-Language Guide

**A chatbot that answers questions about U.S. government contracting rules — and shows you where each answer comes from.**

This guide explains what the project is and how to use it, even if you've never heard of the tools or terms involved. It's a simpler companion to the main [README.md](README.md), which has the full technical detail.

---

## What is this, in one sentence?

It's a question-and-answer app, like a chatbot, that knows the official rules for selling goods and services to the U.S. government — and backs up every answer with a link to the exact rule it used.

---

## Why would anyone want this?

Companies that sell to the U.S. government have to follow a huge rulebook. Two of them, really:

- **The FAR** (Federal Acquisition Regulation) — the main rulebook for government buying.
- **The DFARS** (Defense Federal Acquisition Regulation Supplement) — extra rules for the Department of Defense.

Together these run to thousands of pages, and they change often. Reading them to find one answer is slow and frustrating.

This app lets you **ask a plain question** — like "What does FAR 15.404 say about price analysis?" — and get a **short, accurate answer with citations**, in seconds. It's useful if you:

- Write proposals to win government contracts
- Check that your company follows the rules
- Manage a contract you already have
- Are deciding whether to enter the government market at all

---

## The one idea that makes it work: "RAG"

The app is built on a method called **RAG**, short for **Retrieval-Augmented Generation**. That sounds complex, but the idea is simple.

Think of an open-book exam. A regular AI chatbot answers from memory, and it can confidently make things up. A RAG chatbot is different: before it answers, it **looks up the relevant pages in the rulebook**, then writes its answer using only what it found. "Retrieval" means looking up the right pages. "Generation" means writing the answer. Putting them together keeps the answer grounded in real rules — and lets the app show you exactly which sections it used.

So every answer comes with **citations**: pointers to the exact FAR or DFARS section behind it. You don't have to take the app's word for it.

---

## What's special about how it runs

**It runs entirely on a normal laptop.** No expensive graphics card, no internet service, no monthly fee. Everything happens on your own computer:

- It does **not** call out to a paid online AI service.
- It does **not** need a special graphics chip (a "GPU").
- It works on a typical laptop with **16 GB of memory**.

The trade-off is speed: because it uses your laptop's main processor instead of specialized hardware, an answer takes about **10 to 60 seconds**. That's normal here.

---

## How it answers a question, step by step

When you type a question, here's what happens behind the scenes:

1. **It checks a memory of past answers first.** If you've asked the exact same question before, it replays the saved answer instantly (in about a tenth of a second). This shortcut is called the **answer cache**.
2. **If it's a new question, it searches the rulebook** to find the most relevant sections.
3. **It feeds those sections to a small AI model** (a program called Phi-4, made by Microsoft) that runs on your laptop.
4. **The AI writes the answer**, word by word, so you see it appear as it's typed.
5. **It shows the citations** — the exact rule sections it used.

---

## Three ways it can search the rulebook

The app can look things up in three different ways. You pick one with a setting called `RAG_MODE`. You don't need to understand the inner workings — here's the plain version:

| Mode name | What it's good at | Speed |
|---|---|---|
| **naive** | The basic, reliable search. Finds sections by overall meaning. | Fastest |
| **hybrid** | Combines meaning-based search with exact word matching, so it also catches specific rule numbers and defined terms. | Almost as fast |
| **graph** | Builds a web of how rules and concepts connect, then answers from that web. More powerful, but slower and set up only for one part of the FAR so far. | Slowest |

There's also an optional helper called a **reranker** that double-checks the search results and keeps only the most relevant ones. It's turned on by default and makes answers sharper. Think of it as a second pass that re-sorts the results so the best matches rise to the top.

Most people can leave all the defaults alone. "naive" mode with the reranker is a solid starting point.

---

## What you need before you start

You don't need to be a programmer to follow the setup, but you will be typing commands into a terminal (the text-based control window on your computer). Here's the short version of the requirements:

| You need | Why |
|---|---|
| A computer with **16 GB of memory (RAM)** | The AI model needs room to load. |
| About **8 GB of free disk space** | For the model, the rulebook data, and supporting files. |
| **Windows, Mac, or Linux** | All three work. |
| **Python 3.10–3.12** | The programming language the app is written in. (Version 3.12 is best.) |
| **Git and "Git LFS"** | Tools for downloading the project and its large data files. |
| **Docker** (optional) | A tool that runs the app in a tidy, self-contained box. Only needed for the one-command setup option. |

A "GPU" (a special graphics chip) is **not** required.

---

## Setting it up

These steps download the project, install what it needs, and get the AI model. Type each block into your terminal.

### 1. Download the project

```bash
git clone https://github.com/PWDevens/fedacq-rag-chatbot.git
cd fedacq-rag-chatbot
```

### 2. Turn on Git LFS and pull the large files

"Git LFS" (Large File Storage) handles the big data files, like the pre-built rulebook search index.

```bash
git lfs install
git lfs pull
```

### 3. Create a clean workspace for the project's tools

This makes a private folder (a "virtual environment") so the project's tools don't clash with anything else on your computer.

On Mac or Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install the project and its parts

```bash
pip install -e .
pip install -r requirements.txt
```

### 5. Download the AI model

This pulls the Microsoft Phi-4 model that writes the answers. It's about 4.6 GB, so it may take a while.

```bash
huggingface-cli download microsoft/Phi-4-mini-instruct-onnx \
  --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* \
  --local-dir .
```

That's it. You don't need to build anything else — the searchable rulebook comes pre-built with the download.

---

## Running the app

You have two options. Both end with you opening the app in your web browser at **http://localhost:7860**.

### Option A — The one-command way (using Docker)

If you have Docker installed, this is the simplest path:

```bash
cd docker
docker compose up --build
```

Then open **http://localhost:7860** in your browser. To stop it, run `docker compose down`.

### Option B — Running it directly with Python

On Windows, this command is the easiest:

```bash
python -m flask --app src.app run --host=0.0.0.0 --port=7860
```

On Mac or Linux, you can use this instead:

```bash
hypercorn --bind 0.0.0.0:7860 src.app.asgi:app
```

Then open **http://localhost:7860**.

**A note on the first answer:** the very first question after you start the app takes 15–30 seconds longer than usual, because the app is loading the AI model into memory. After that, answers come faster. This is normal.

---

## Asking a question

Once the app is running, you can use the web page in your browser. Type a question like:

> What does FAR 15.404 say about price analysis?

You'll get back a short summary, the citations (which rule sections were used), and the AI's explanation.

If you'd rather send a question from the terminal instead of the web page, you can:

```bash
curl -X POST http://localhost:7860/chat_stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does FAR 15.404 say about price analysis?"}'
```

---

## Why answers sometimes repeat instantly

The app remembers questions it has already answered and saves the result in a small file. The next time someone asks the **exact same question**, it replays the saved answer in about a tenth of a second instead of taking tens of seconds to think it through again. This memory survives even if you restart the app. You can turn it off, but there's rarely a reason to.

---

## Adjusting how it behaves (optional)

The app works fine out of the box. But if you want to change something, you do it through **environment variables** — named settings you can set before starting the app. You don't edit any code. A few useful ones:

| Setting | What it controls |
|---|---|
| `RAG_MODE` | Which search method to use: `naive`, `hybrid`, or `graph`. |
| `RERANK` | Whether the result-sorting helper is on (it's on by default). |
| `MAX_NEW_TOKENS` | How long answers can be. Smaller = faster. |
| `ANSWER_CACHE` | Whether to remember and replay past answers (on by default). |

The full list is in the main [README.md](README.md).

---

## A few terms, in plain words

You'll see these in the main README. Here's what they really mean:

- **FAR / DFARS** — The official rulebooks for selling to the U.S. government (the second one is for defense). The text is public and free to use.
- **RAG** — The "look it up, then answer" method that keeps the chatbot honest. (See the section above.)
- **Citation** — A pointer to the exact rule the answer came from.
- **Phi-4** — The small Microsoft AI model, running on your laptop, that writes the answers.
- **ChromaDB** — The searchable filing cabinet that stores the rulebook so the app can find sections fast.
- **Embedding** — A way of turning text into numbers so a computer can measure which passages mean similar things. It's how meaning-based search works under the hood.
- **Token** — A small piece of a word. AI models read and write in tokens, which is why answers appear a few pieces at a time.
- **Reranker** — A second-pass helper that re-sorts search results so the best matches come first.
- **Answer cache** — The app's memory of past answers, used to reply to repeat questions instantly.
- **GPU** — A special graphics chip that speeds up AI. This app is built so you **don't** need one.
- **Docker** — A tool that runs the whole app in a self-contained box, so setup is one command.
- **Terminal** — The text-based window where you type commands to your computer.
- **Environment variable** — A named setting you can change without touching any code.

---

## Where things live (for the curious)

If you open the project folder, the most useful parts are:

- **`src/`** — The actual app code.
- **`data/`** — The pre-built searchable rulebook and related data.
- **`tests/`** — Automated checks that make sure the app works.
- **`docker/`** — Files for the one-command Docker setup.
- **`README.md`** — The full, detailed guide.

---

## Cost, licensing, and credit

- The app is **free and open source** under the MIT License (a permissive, no-cost license).
- The **FAR and DFARS text** is a work of the U.S. government and is in the public domain — free for anyone to use.
- The **Phi-4 AI model** is made by Microsoft and used under Microsoft's own license. You download it directly from Microsoft's source; it isn't bundled here.

---

*This is a simplified guide. For complete technical detail — including how to rebuild the rulebook index, run the advanced "graph" mode, and tune performance — see the main [README.md](README.md).*
