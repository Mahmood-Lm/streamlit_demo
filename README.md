# Streamlit Course Recommender System

A machine learning-powered course recommendation system built with Streamlit and Python. This application provides personalized course recommendations using multiple algorithms including Course Similarity and K-Nearest Neighbors (KNN).

## Features

- **Interactive Web Interface**: Built with Streamlit for easy-to-use course selection and recommendation viewing
- **Multiple Recommendation Algorithms**:
  - Course Similarity: Based on content similarity between courses
  - K-Nearest Neighbors (KNN): Collaborative filtering using user behavior patterns
  - Additional models ready for implementation (User Profile, Clustering, NMF, Neural Networks, etc.)
- **Customizable Parameters**: Tune hyperparameters through interactive sliders
- **Real-time Recommendations**: Get instant course suggestions based on your selections

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/streamlit_demo.git
cd streamlit_demo
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run recommender_app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Select courses you have completed from the interactive table

4. Choose a recommendation model from the sidebar:
   - **Course Similarity**: Recommends courses similar to ones you've taken
   - **KNN**: Finds users with similar preferences and recommends their courses

5. Tune hyperparameters using the sliders:
   - For Course Similarity: Top courses count and similarity threshold
   - For KNN: Number of neighbors (K) and similarity threshold

6. Click "Train Model" (if required) and then "Recommend New Courses"

## Files Description

- `recommender_app.py`: Main Streamlit application interface
- `backend.py`: Core recommendation algorithms and data processing
- `requirements.txt`: Python package dependencies
- `ratings.csv`: User-item rating data
- `course_processed.csv`: Course information and metadata
- `sim.csv`: Pre-computed course similarity matrix
- `courses_bows.csv`: Course content features (bag-of-words)

## Algorithms

### Course Similarity
Uses content-based filtering to recommend courses similar to those you've already taken. Calculates similarity scores based on course features and content.

### K-Nearest Neighbors (KNN)
Implements collaborative filtering by:
1. Finding users with similar rating patterns
2. Using cosine similarity to identify neighbors
3. Predicting ratings for unrated courses based on neighbor preferences
4. Recommending top-scoring courses above the threshold

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Streamlit-AgGrid**: Interactive data tables

## Contributing

Feel free to contribute to this project by:
1. Adding new recommendation algorithms
2. Improving the user interface
3. Enhancing data processing capabilities
4. Adding evaluation metrics

## License

This project is open source and available under the MIT License.