# studydesk
StudyDesk is a Chatbot which will read your pdf notes and answer questions related to the notes and, summarize topics and generates a quiz to test your understanding
Live demo: https://studydesk-frgwavivylkq4mgerkijxi.streamlit.app/

StudyDesk lets you upload your lecture notes or slides as PDFs and then ask questions about them, get a revision summary of a topic, or generate a multiple-choice quiz. Every answer is built only from the pages you uploaded and comes with page numbers, so you can check anything that looks off.
I built it to revise from my own course slides. A general chatbot tends to answer from the whole internet, and I wanted something that sticks to what my lecturer actually taught.

How it works:
The PDF text is pulled out with PyMuPDF. Slide decks are messy, so I strip out headers and footers that repeat on every page, plus stray symbols. Pages that are just images (no text layer) are read with Tesseract OCR.
The cleaned text is cut into chunks of 800 characters with 150 characters of overlap, and each chunk remembers which file and page it came from. The chunks are turned into vectors with the all-MiniLM-L6-v2 model and stored in a FAISS index.
When you ask something, the question is embedded the same way, the closest few chunks are fetched, and they are sent to Gemini with the instruction to answer only from those notes and say so if the answer isn't there. The summary and quiz tabs work the same way, just with a different prompt.

Things that don't work well yet:
It only reads PDFs. Text inside diagrams and equations that are saved as pictures can come out garbled or get missed.
OCR quality depends on the slides. Small or low-contrast text gets misread sometimes.
Search is by meaning only, so a question that needs information from many different slides at once can miss some of them.
The quiz questions are written by the model, so give the answers a quick sanity check.
If you see a "high demand" error, that is Google's Gemini service being overloaded, not a bug in the app. Trying again a bit later usually works. There is no automatic retry yet.
