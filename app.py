import streamlit as st
import arxiv
import pandas as pd
from transformers import pipeline

def fetch_papers(query, max_results):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for result in search.results():
        papers.append({
            'published': result.published,
            'title': result.title,
            'abstract': result.summary,
            'categories': result.categories
        })
    return papers

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']




st.title("ArXiv Paper Summarizer")

query = st.text_input("Enter your search query:", "ai OR artificial intelligence OR machine learning")
max_results = st.slider("Number of results", 1, 20, 10)

if st.button("Fetch Papers"):
    with st.spinner("Fetching papers..."):
        papers = fetch_papers(query, max_results)
        if papers:
            df = pd.DataFrame(papers)
            st.session_state['papers'] = df
            st.success("Papers retrieved successfully!")
            st.dataframe(df[['title', 'published', 'categories']])
        else:
            st.warning("No papers found.")

if 'papers' in st.session_state:
    df = st.session_state['papers']
    paper_index = st.selectbox("Select a paper to summarize:", df.index, format_func=lambda i: df['title'][i])
    abstract = df['abstract'][paper_index]
    
    if st.button("Summarize Abstract"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(abstract)
            st.subheader("Summary:")
            st.write(summary)


