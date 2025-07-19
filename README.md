# Overview

Emeowtions is a React + Flask full-stack Audio Classification application that analyses cat emotions through their sounds
in order to help humans to communicate better with their furry friends. Our software is capable of analyzing cat emotions in
uploaded audio (.mp3/.wav) files, video (.mp4) files, and even live audio recording through microphones. Emeowtions
utilizes a custom Acoustic Emotion Recognition (AER) AI model that uses a Deep Neural Network (DNN) algorithm to
recognize emotions in cat sounds. The accuracy of our emotion analysis/classification model has also proven to be
quite accurate in testing, displaying an **accuracy level and weighted average f1-score of 75% and 74% respectively.**
To further help users to understand their cats' emotions, Emeowtions has a built-in texting-based AI assistant that can
aid them by providing detailed and intricate explanations or instructions. We hope that this software will be able to
aid humans, especially cat owners, to understand cats better.

# Installation Guide

This guide is for **installing Emeowtions locally**. We highly recommend to access the deployed version which
is accessible through the link included in this repository. The emotion analysis may take some time to finish since the app
needs to load resources it needs to execute the analysis process.

**Note: Emeowtion's source code files don't include GEMINI API Keys for security reasons, therefore please use your own API key
if you want to use our application's AI Assistant feature locally.**

## Requirements
* Python Version 3.10.13 (Highly recommend to use virtual environments such as .conda and .venv)
* Python Libraries (See requirements.txt in the backend folder)

## Instructions
1. Ensure to install all the required software listed in the **Requirements** section
2. Download Emeowtions by cloning the repository or downloading the app in ZIP file
3. Go to the backend folder through your computer's terminal
4. Type "python cat_emotion_analyzer.py" in your terminal and press Enter
5. Wait until "Running on http://127.0.0.1:5000" message appear in your terminal
6. Access http://127.0.0.1:5000 with your preferred browser
