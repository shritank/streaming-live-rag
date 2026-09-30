# Streaming Live RAG - architecture diagram (Mermaid)

Renders on GitHub and in the Mermaid live editor (https://mermaid.live): copy everything between the fences.
Component names match the code (`streaming_rag/...`); the design rationale is in `docs/architecture_brief.docx`.

```mermaid
flowchart TB
    %% ---------- input ----------
    subgraph IN["Input (streaming)"]
        direction LR
        T["Transcript chunks<br/>+ utterance_end"]
        A["Audio front end<br/>(optional, out of scope)<br/>Whisper, asr/"]
        A -.-> T
    end

    %% ---------- 1 controller ----------
    subgraph CTL["1. Stream controller (controller/)"]
        direction LR
        ST["Intent stability<br/>anchors, closure, growth"]
        POL{"Policy"}
        DEC["Multi-intent decomposition<br/>+ supersession of growing clauses"]
        W["WAIT"]
        SU["SUPPRESS<br/>(reformat / chit-chat:<br/>no retrieval)"]
        ST --> POL
        POL -->|"stable partial: provisional<br/>compound: multi_intent<br/>late detail: refinement"| DEC
        POL -->|"unstable"| W
        POL -->|"presentation-only"| SU
    end

    %% ---------- 2 retrieval ----------
    subgraph RET["2. Hybrid retrieval (retrieval/), runs during speech"]
        direction LR
        BM["BM25 (sparse)"]
        DN["e5-small-v2<br/>(dense, ONNX)"]
        RRF["Reciprocal-rank fusion<br/>+ dedup, top 20"]
        CE["Cross-encoder rerank<br/>MiniLM (ONNX)"]
        BM --> RRF
        DN --> RRF
        RRF --> CE
    end

    %% ---------- 3 synthesis ----------
    subgraph SYN["3. Session synthesis (session/)"]
        direction LR
        CS["Claim selection<br/>verbatim corpus sentence"]
        RG{"Refusal gate<br/>learned P(correct) >= 0.40"}
        GR["Grounding check<br/>citation = Doc_ID section"]
        UN["Uncertainty note<br/>(refuse, don't guess)"]
        VER["Answer version<br/>new / refine / restructure"]
        CS --> RG
        RG -->|"supported"| GR
        RG -->|"not supported"| UN
        GR --> VER
        UN --> VER
    end

    %% ---------- output, stores, telemetry, hardware ----------
    REC["Structured output record<br/>retrieval_events, sub_queries, answer, citations, uncertainty"]
    CORP[("Corpus<br/>chunks, BM25 index, e5 embeddings")]
    SESS[("Session store<br/>answer versions per (session_id, utterance_id)")]
    TEL["Telemetry (JSONL)<br/>decisions, retrievals, versions, latency, cost"]
    GPU["ONNX Runtime on CUDA<br/>(mandatory when an NVIDIA GPU is present;<br/>CPU only by explicit opt-out)"]

    %% ---------- main flow (stage to stage) ----------
    IN --> CTL
    CTL -->|"sub-queries"| RET
    RET -->|"ranked evidence"| SYN
    SYN --> REC

    %% ---------- side flows ----------
    CORP --- RET
    SESS --- SYN
    GPU -.-> RET
    CTL -.-> TEL
    RET -.-> TEL
    SYN -.-> TEL

    classDef opt stroke-dasharray: 5 5,fill:#f5f5f5,color:#555;
    classDef store fill:#eef4ff,stroke:#4a6fa5;
    classDef out fill:#eefaf0,stroke:#3c8d55;
    class A opt;
    class CORP,SESS store;
    class REC,TEL out;
```
