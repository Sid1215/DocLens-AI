import streamlit as st

from pathlib import Path

from rag import (
    load_document,
    create_embeddings,
    create_vector_store,
    search_documents,
    create_llm,
    generate_answer,
    generate_comparison
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocLens AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CSS
# ============================================================

css_path = Path("styles/style.css")

with open(css_path, "r") as css_file:
    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Doc<span class="brand-accent">Lens</span> AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">Intelligent document analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Workspace</div>',
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Workspace",
        [
            "💬 Ask Documents",
            "🔄 Compare Documents"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">About</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-info">'
        'DocLens AI uses Retrieval-Augmented Generation '
        'to retrieve information from your documents '
        'and generate grounded responses.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">Technology</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-info">'
        'LangChain<br>'
        'FAISS<br>'
        'HuggingFace Embeddings<br>'
        'Groq / Qwen'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">'
    'Understand your <span class="hero-accent">documents.</span>'
    '</div>'
    '<div class="hero-description">'
    'Upload your documents and use AI to extract answers, '
    'compare information, and discover useful insights.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DOCUMENT SECTION
# ============================================================

st.markdown(
    '<div class="section-label">Workspace</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">📂 Your Documents</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD CARD
# ============================================================

st.markdown(
    '<div class="upload-card">'
    '<div class="upload-title">Upload your PDF documents</div>'
    '<div class="upload-description">'
    'Add one or more PDF files to start analyzing them.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# ============================================================
# EMPTY STATE
# ============================================================

if not uploaded_files:

    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">📄</div>'
        '<div class="empty-title">No documents yet</div>'
        '<div class="empty-description">'
        'Upload one or more PDF files above. '
        'Your documents will appear here once uploaded.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

else:

    document_count = len(uploaded_files)

    st.markdown(
        f'<div class="status">'
        f'✓ {document_count} document'
        f'{"s" if document_count != 1 else ""} '
        f'uploaded successfully'
        f'</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------------

    for uploaded_file in uploaded_files:

        st.markdown(
            f'<div class="document-card">'
            f'<span class="document-icon">📄</span>'
            f'<span class="document-name">'
            f'{uploaded_file.name}'
            f'</span>'
            f'<div class="document-type">PDF document</div>'
            f'</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    all_chunks = []


    for uploaded_file in uploaded_files:

        file_path = f"data/{uploaded_file.name}"


        with open(file_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )


        chunks = load_document(
            file_path
        )


        all_chunks.extend(
            chunks
        )


    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embeddings = create_embeddings()


    # --------------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------------

    vector_store = create_vector_store(
        all_chunks,
        embeddings
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = create_llm()


    # ========================================================
    # ASK DOCUMENTS
    # ========================================================

    if mode == "💬 Ask Documents":

        st.markdown(
            '<div class="section-label">Mode</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">💬 Ask Documents</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">'
            'Ask anything about your documents'
            '</div>'
            '<div class="mode-description">'
            'DocLens retrieves relevant information from your '
            'documents and generates a grounded answer.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


        query = st.text_input(
            "Question",
            placeholder="What would you like to know?",
            label_visibility="collapsed"
        )


        if query:

            with st.spinner(
                "Searching your documents..."
            ):

                results = search_documents(
                    vector_store,
                    query
                )


                answer = generate_answer(
                    llm,
                    query,
                    results
                )


            st.markdown(
                '<div class="section-label">Response</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">✨ Answer</div>',
                unsafe_allow_html=True
            )

            st.markdown(answer)


    # ========================================================
    # COMPARE DOCUMENTS
    # ========================================================

    elif mode == "🔄 Compare Documents":

        st.markdown(
            '<div class="section-label">Mode</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🔄 Compare Documents</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="mode-card">'
            '<div class="mode-title">'
            'Compare your documents'
            '</div>'
            '<div class="mode-description">'
            'Analyze skills, experience, projects, education, '
            'strengths, gaps, and other relevant information '
            'across your documents.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


        if len(uploaded_files) < 2:

            st.warning(
                "Upload at least two documents to compare them."
            )


        else:

            st.markdown(
                f'<div class="status">'
                f'✓ Ready to compare '
                f'{len(uploaded_files)} documents'
                f'</div>',
                unsafe_allow_html=True
            )


            if st.button(
                "🔍 Analyze Documents",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing your documents..."
                ):

                    comparison = generate_comparison(
                        llm,
                        all_chunks
                    )


                st.markdown(
                    '<div class="section-label">Analysis</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-title">'
                    '📊 Comparison Results'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(comparison)
