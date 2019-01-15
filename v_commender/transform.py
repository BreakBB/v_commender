import numpy as np

genre_count = 43
director_count = 4205
actor_count = 13748
writer_count = 6685


def movie_to_vector(movie, norm=False):
    if norm:
        return [
            round(movie['genre_ids'][0] / genre_count, 6),
            round(movie['director_ids'][0] / director_count, 6),
            round(movie['actor_ids'][0] / actor_count, 6),
            round(movie['actor_ids'][1] / actor_count, 6),
            round(movie['actor_ids'][2] / actor_count, 6)
            # round(movie['writer_ids'][0] / writer_count, 6),
            # round(movie['writer_ids'][1] / writer_count, 6)
        ]
    else:
        return [
            movie['genre_ids'][0],
            movie['director_ids'][0],
            movie['actor_ids'][0],
            movie['actor_ids'][1],
            movie['actor_ids'][2]
            # movie['writer_ids'][0],
            # movie['writer_ids'][1]
        ]


def movies_to_list(movies, norm=False):
    movie_list = []
    for movie in movies:
        movie_list.append(movie_to_vector(movie, norm))

    return movie_list


def create_training_data(data, norm=False):
    # print("Generating trainings data from movies")
    training_movies = []
    training_votes = []

    for row in data:
        if row['vote'] is True:
            training_votes.append([1])
        else:
            training_votes.append([0])

        # Add 1 genre, 1 director, 3 actors and 2 writers
        if norm:
            # Add 1 genre, 1 director, 3 actors and 2 writers
            movie_vector = [
                round(row['genre_ids'][0] / genre_count, 6),
                round(row['director_ids'][0] / director_count, 6),
                round(row['actor_ids'][0] / actor_count, 6),
                round(row['actor_ids'][1] / actor_count, 6),
                round(row['actor_ids'][2] / actor_count, 6)
                # round(row['writer_ids'][0] / writer_count, 6),
                # round(row['writer_ids'][1] / writer_count, 6)
            ]
        else:
            movie_vector = [
                row['genre_ids'][0],
                row['director_ids'][0],
                row['actor_ids'][0],
                row['actor_ids'][1],
                row['actor_ids'][2]
                # row['writer_ids'][0],
                # row['writer_ids'][1]
            ]

        training_movies.append(movie_vector)

    return {
        "movies": np.array(training_movies),
        "votes": np.array(training_votes)
    }
