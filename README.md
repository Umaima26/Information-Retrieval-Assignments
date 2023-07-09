# Information-Retrieval-Assignments
This repository includes my academic assignments that I created while studying the elective "Information Retrieval" at university. The assignments focus on implementing Information Retrieval (IR) techniques and models to retrieve relevant information from a collection of Short Stories.

# Assignment 1: Inverted Index and Boolean Retrieval Model
Assignment 1 provides an opportunity to delve into the concepts of inverted index and the Boolean Retrieval Model. By implementing an inverted index and positional index, we can efficiently determine which documents contain specific terms and their proximity. The pre-processing pipeline, including tokenization, case folding, stop-word removal, and stemming, helps us prepare the documents for indexing. With a simplified Boolean query processing routine, we can execute queries combining terms using Boolean operators (AND, OR, and NOT).

# Assignment 2: Vector Space Model (VSM) for Information Retrieval
Assignment 2 focuses on the Vector Space Model (VSM) for information retrieval. By building a vector space of features using tf*idf-based weighting, we can represent documents and queries in the same feature space. Cosine similarity allows us to compute the relevance scores between documents and queries. The pre-processing pipeline, including tokenization, case folding, stop-word removal, and lemmatization, helps us prepare the documents and queries for vector space representation. The implementation involves placing queries in the feature space and computing scores based on cosine similarity.

# Repository Structure
The repository is structured as follows:

**ShortStories:** A directory containing the collection of short stories used in both assignments.
**Assignment 1 & Assignment 2:** Directories containing the materials related to the respective assignments, including the problem statement, my solution, some test queries, and relevant text files that are generated when you run the program.
Each assignment directory may include additional files or subdirectories, depending on the specific requirements and implementation details of the assignment. The relevant text files generated during the program execution are stored within their respective assignment directories.
Feel free to explore the contents of each directory to gain a better understanding of the assignments and access the necessary files for running the programs and evaluating the results.
Please refer to the individual assignment directories for more specific instructions and guidelines related to each assignment.


# User Interface (UI)
The user interface for both assignments is built using PyQt5 Designer, providing an intuitive graphical representation of the working models. The integration of PyQt5 Designer allows for enhanced functionality, including support for free text query search and phrase query search.
