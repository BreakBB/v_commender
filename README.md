# v_commender

This project allows classification and prediction of movie recommendations. It includes:

- A wrapper for the Gaussian Naive Bayes classifier from [sklearn](https://scikit-learn.org/stable/)
- A rudimentary implementation of a Neural Network with one hidden layer and a flexible amount of neurones

## Requirements

- Install the python dependencies using the **requirements.txt** with:

    ```pip install -r requirements.txt```

- A **data.json** file with the following structure:

```JSON
{
  "voted": [
    {
      "vote": true,
      "title": "Avengers: Infinity War",
      "genre_ids": [
        1,
        2
      ],
      "actor_ids": [
        2321,
        2333,
        8679,
        11056
      ],
      "writer_ids": [
        626,
        1085,
        1674,
        2129,
        2627,
        3042,
        3115,
        3585,
        3739,
        5926,
        5975,
        6000,
        6003
      ],
      "director_ids": [
        274,
        1910
      ],
      ...
    }
  ],
  "predict": [
    {
      "movie_id": "B00A7CF1QY",
      "genre_ids": [
        1,
        12,
        37
      ],
      "actor_ids": [
        3776,
        6078,
        11720,
        12173
      ],
      "writer_ids": [
        1266,
        5392,
        6351
      ],
      "director_ids": [
        3980
      ]
    },
    ...
  ]  
}
```

The **genre_ids**, **actor_ids**, **writer_ids** and **director_ids** are simple mappings of a name to an id. Amazon Prime Video assigns an id to each movie/series which look like **movie_id**. The **vote** field is a boolean with true if the user likes this movie/series or false if he dislikes it.

## Usage

1. Make sure to have a data.json in the main directory
2. Call the main.py with one of the following arguments:
    - "bayes" for the Naive Bayes classifier
    - "neural" to train and predict with the neural network
    - or any non empty argument to calculate metrics for both algorithms
