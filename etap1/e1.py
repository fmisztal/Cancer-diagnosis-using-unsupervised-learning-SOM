
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("../breast_cancer_wisconsin/data.csv")

    # df.hist( column='radius_mean', bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu radius_mean')
    # plt.show()
    #
    # df.hist(column="texture_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu texture_mean')
    # plt.show()
    #
    # df.hist(column="perimeter_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu perimeter_mean')
    # plt.show()
    #
    # df.hist(column="area_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu area_mean')
    # plt.show()
    #
    # df.hist(column="smoothness_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu smoothness_mean')
    # plt.show()
    #
    # df.hist(column="compactness_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu compactness_mean')
    # plt.show()
    #
    # df.hist(column="concavity_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu concavity_mean')
    # plt.show()
    #
    # df.hist(column="concave points_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu concave points_mean')
    # plt.show()
    #
    # df.hist(column="symmetry_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu symetry_mean')
    # plt.show()
    #
    # df.hist(column="fractal_dimension_mean", bins=100)
    # plt.xlabel('Wartości')
    # plt.ylabel('Liczba wystąpień')
    # plt.title('Histogram atrybutu fractal_dimension_mean')
    # plt.show()

    df_B = df.loc[df['diagnosis'] == 'B']
    df_M = df.loc[df['diagnosis'] == 'M']

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['radius_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu radius_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['radius_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu radius_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')


    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['texture_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu texture_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['texture_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu texture_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['perimeter_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu perimeter_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['perimeter_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu perimeter_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')


    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['area_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu area_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['area_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu area_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['smoothness_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu smoothness_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['smoothness_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu smoothness_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['compactness_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu compactness_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['compactness_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu compactness_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['concavity_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu concavity_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['concavity_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu concavity_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['concave points_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu concave points_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['concave points_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu concave points_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['symmetry_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu symmetry_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['symmetry_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu symmetry_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    df_B['fractal_dimension_mean'].hist(ax=ax1, bins=50)
    ax1.set_title('Histogram atrybutu fractal_dimension_mean dla diagnozy B')
    ax1.set_xlabel('Wartości')
    ax1.set_ylabel('Liczba wystąpień')

    df_M['fractal_dimension_mean'].hist(ax=ax2, bins=50)
    ax2.set_title('Histogram atrybutu fractal_dimension_mean dla diagnozy M')
    ax2.set_xlabel('Wartości')
    ax2.set_ylabel('Liczba wystąpień')


    plt.show()