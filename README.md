\# Student Performance Prediction MLOps Pipeline



\## Project Overview



The \*\*Student Performance Prediction MLOps Pipeline\*\* is an end-to-end machine learning project that predicts a student's final academic score using academic, attendance, behavioral, and demographic features.



The project demonstrates the complete machine learning lifecycle, including data preprocessing, model training, experiment tracking, data versioning, model serving, deployment, testing, monitoring, and automated workflows.



The project is designed using MLOps principles to make the machine learning workflow reproducible, maintainable, and deployable.



\---



\## Objectives



The main objectives of this project are:



\- Predict student final performance using machine learning.

\- Perform data preprocessing and feature engineering.

\- Train and compare multiple machine learning models.

\- Track experiments using MLflow.

\- Version the dataset using DVC.

\- Automatically select the best-performing model.

\- Serve predictions through a FastAPI REST API.

\- Provide an interactive Streamlit dashboard.

\- Containerize the prediction service using Docker.

\- Implement automated testing using Pytest.

\- Implement CI using GitHub Actions.

\- Support automatic model retraining.

\- Perform basic model performance monitoring.



\---



\## Dataset



The dataset contains 1000 student records and includes academic, attendance, behavioral, and demographic information.



\### Features



| Feature | Description |

|---|---|

| gender | Student gender |

| age | Student age |

| study\_hours | Average daily study hours |

| attendance | Attendance percentage |

| previous\_score | Previous academic score |

| assignments\_completed | Number of completed assignments |

| sleep\_hours | Average sleep duration |

| extracurricular | Participation in extracurricular activities |

| internet\_access | Internet access availability |

| parental\_support | Level of parental support |

| final\_score | Final academic score - target variable |



\### Target Variable



`final\_score`



The project treats student performance prediction as a regression problem.



\---



\## Machine Learning Models



Three regression models are trained and compared:



1\. Linear Regression

2\. Random Forest Regressor

3\. Gradient Boosting Regressor



The models are evaluated using:



\- RMSE

\- MAE

\- R² Score



The model with the best R² performance is automatically selected as the best model.



\---



\## MLOps Architecture



```text

&#x20;                 Student Dataset

&#x20;                       |

&#x20;                       v

&#x20;                      DVC

&#x20;                Data Versioning

&#x20;                       |

&#x20;                       v

&#x20;                Data Preprocessing

&#x20;                       |

&#x20;                       v

&#x20;                Feature Engineering

&#x20;                       |

&#x20;                       v

&#x20;                 Train/Test Split

&#x20;                       |

&#x20;         +-------------+-------------+

&#x20;         |             |             |

&#x20;         v             v             v

&#x20;    Linear          Random        Gradient

&#x20;   Regression       Forest        Boosting

&#x20;         |             |             |

&#x20;         +-------------+-------------+

&#x20;                       |

&#x20;                       v

&#x20;                    MLflow

&#x20;               Experiment Tracking

&#x20;                       |

&#x20;                       v

&#x20;               Model Comparison

&#x20;                       |

&#x20;                       v

&#x20;                 Best Model

&#x20;                       |

&#x20;              +--------+--------+

&#x20;              |                 |

&#x20;              v                 v

&#x20;           FastAPI          Streamlit

&#x20;              |                 |

&#x20;              +--------+--------+

&#x20;                       |

&#x20;                       v

&#x20;                 User Prediction



&#x20;      GitHub --> GitHub Actions --> CI Testing



&#x20;      Docker --> Containerized FastAPI Service



&#x20;      Retraining --> Evaluation --> Monitoring

