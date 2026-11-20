import streamlit as st
from rag import ocr_available, load_pdf, chunk_pages, NotesIndex, answer_question, summarize_topic, generate_quiz

CHUNK_SIZE = 800   # best-performing setting from evaluate.py; no need to expose it to users
OVERLAP = 150
DEPTH = {"Focused": 3, "Balanced": 4, "Broad": 6}  # how many passages are pulled from the notes

st.set_page_config(page_title="StudyDesk", layout="centered")

st.title("StudyDesk")
st.markdown(
    "Upload your lecture notes or slides as PDFs, then ask questions, get a quick "
    "revision summary, or test yourself with a practice quiz. "
    "Answers come only from your own notes, with page numbers so you can check them."
)

files = st.file_uploader("Your notes (PDF)", type="pdf", accept_multiple_files=True)

if not files:
    st.caption("Add one or more PDFs to get started.")
    st.stop()

# Build the search index once per set of files
key = tuple((f.name, f.size) for f in files)
if st.session_state.get("key") != key:
    with st.spinner("Reading your notes. Slides that are only images can take a minute the first time."):
        pages = []
        for f in files:
            pages += load_pdf(f.getvalue(), f.name)
        chunks = chunk_pages(pages, CHUNK_SIZE, OVERLAP)
        if not chunks:
            st.error("I couldn't find any readable text in these files.")
            st.stop()
        st.session_state["index"] = NotesIndex(chunks)
        st.session_state["key"] = key
        st.session_state["n_pages"] = len({(p["source"], p["page"]) for p in pages})

index = st.session_state["index"]
st.caption(f"Ready: {st.session_state['n_pages']} pages read from {len(files)} file(s).")

if not ocr_available():
    st.warning("Text recognition isn't available here, so slides that are only images will be skipped.")

with st.expander("Options"):
    depth = st.select_slider(
        "How much of your notes to search for each answer",
        options=list(DEPTH),
        value="Balanced",
        help="Focused is faster and sticks to the closest match. Broad pulls in more pages, "
             "which helps with questions that span several topics.",
    )
k = DEPTH[depth]


def show_sources(hits):
    with st.expander("Where this came from"):
        for h in hits:
            st.markdown(f"**{h['source']}, page {h['page']}**")
            st.caption(h["text"][:400] + ("..." if len(h["text"]) > 400 else ""))


def run(fn, spinner_text, *args):
    """Call the pipeline and show a readable message if something goes wrong."""
    try:
        with st.spinner(spinner_text):
            return fn(*args)
    except KeyError:
        st.error("The Gemini API key is missing. Add GEMINI_API_KEY to the app's secrets.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    return None


tab_ask, tab_sum, tab_quiz = st.tabs(["Ask a question", "Summarize a topic", "Practice quiz"])

with tab_ask:
    with st.form("ask_form", border=False):
        q = st.text_input("Your question", placeholder="e.g. What is a core point in DBSCAN?")
        go = st.form_submit_button("Answer")
    if go and q.strip():
        out = run(answer_question, "Looking through your notes...", index, q, k)
        if out:
            ans, hits = out
            st.markdown(ans)
            show_sources(hits)

with tab_sum:
    with st.form("sum_form", border=False):
        topic = st.text_input("Topic or chapter", placeholder="e.g. Gaussian mixture models")
        go = st.form_submit_button("Summarize")
    if go and topic.strip():
        out = run(summarize_topic, "Putting a summary together...", index, topic, k + 2)
        if out:
            ans, hits = out
            st.markdown(ans)
            show_sources(hits)

with tab_quiz:
    with st.form("quiz_form", border=False):
        qtopic = st.text_input("Quiz topic", placeholder="e.g. Minimum spanning tree clustering")
        n = st.number_input("Number of questions", 1, 10, 5)
        go = st.form_submit_button("Make quiz")
    if go and qtopic.strip():
        out = run(generate_quiz, "Writing your questions...", index, qtopic, int(n), k + 2)
        if out:
            ans, hits = out
            st.markdown(ans)
            show_sources(hits)
