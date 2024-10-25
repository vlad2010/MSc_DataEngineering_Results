        perror("Error creating thread");
        fclose(wordlist);
        return 1;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&wordlist_mutex);
    fclose(wordlist);
    return 0;
}
