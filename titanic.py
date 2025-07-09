{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNJzOSsOSsc1gp0JYHsoiDg",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AbubakarSiddiqimohammed2005/CODSOFT/blob/main/titanic.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MgGpnVJjhXGH",
        "outputId": "085f466e-21a4-4461-afcd-17d75f0fb333"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Missing values:\n",
            "PassengerId      0\n",
            "Survived         0\n",
            "Pclass           0\n",
            "Name             0\n",
            "Sex              0\n",
            "Age            177\n",
            "SibSp            0\n",
            "Parch            0\n",
            "Ticket           0\n",
            "Fare             0\n",
            "Cabin          687\n",
            "Embarked         2\n",
            "dtype: int64\n",
            "Accuracy: 82.12 %\n",
            "Confusion Matrix:\n",
            "[[92 13]\n",
            " [19 55]]\n",
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.83      0.88      0.85       105\n",
            "           1       0.81      0.74      0.77        74\n",
            "\n",
            "    accuracy                           0.82       179\n",
            "   macro avg       0.82      0.81      0.81       179\n",
            "weighted avg       0.82      0.82      0.82       179\n",
            "\n",
            "Feature Importances:\n",
            "Pclass: 0.087\n",
            "Sex: 0.271\n",
            "Age: 0.25\n",
            "SibSp: 0.054\n",
            "Parch: 0.04\n",
            "Fare: 0.265\n",
            "Embarked: 0.033\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
        "\n",
        "# Load the dataset\n",
        "try:\n",
        "    df = pd.read_csv(\"Titanic-Dataset.csv\")\n",
        "except FileNotFoundError as e:\n",
        "    print(\"❌ Titanic-Dataset.csv not found. Please make sure the file is in the same folder.\")\n",
        "    raise e\n",
        "\n",
        "print(\"Missing values:\")\n",
        "print(df.isnull().sum())\n",
        "\n",
        "df.drop(columns=[\"PassengerId\", \"Name\", \"Ticket\", \"Cabin\"], inplace=True)\n",
        "df[\"Age\"] = df[\"Age\"].fillna(df[\"Age\"].median())\n",
        "df[\"Embarked\"] = df[\"Embarked\"].fillna(df[\"Embarked\"].mode()[0])\n",
        "\n",
        "le = LabelEncoder()\n",
        "df[\"Sex\"] = le.fit_transform(df[\"Sex\"])\n",
        "df[\"Embarked\"] = le.fit_transform(df[\"Embarked\"])\n",
        "\n",
        "X = df.drop(\"Survived\", axis=1)\n",
        "y = df[\"Survived\"]\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "print(\"Accuracy:\", round(accuracy_score(y_test, y_pred) * 100, 2), \"%\")\n",
        "print(\"Confusion Matrix:\")\n",
        "print(confusion_matrix(y_test, y_pred))\n",
        "print(\"Classification Report:\")\n",
        "print(classification_report(y_test, y_pred))\n",
        "\n",
        "importances = model.feature_importances_\n",
        "features = X.columns\n",
        "\n",
        "print(\"Feature Importances:\")\n",
        "for name, score in zip(features, importances):\n",
        "    print(f\"{name}: {round(score, 3)}\")\n"
      ]
    }
  ]
}