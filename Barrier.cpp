#include <iostream>
#include "Barrier.h"

Barrier::Barrier(int numThreads)
        : mutex(PTHREAD_MUTEX_INITIALIZER), cv(PTHREAD_COND_INITIALIZER), count(0),
          numThreads(numThreads)
{
}


Barrier::~Barrier()
{
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cv);
}


void Barrier::barrier()
{
    pthread_mutex_lock(&mutex);
    if (++count < numThreads)
    {
     //   std::cout << "still waiting for everyone" << std::endl;
        pthread_cond_wait(&cv, &mutex);
    }
    else
    {
        count = 0;
      //  std::cout << "everyone arrived" << std::endl;
        pthread_cond_broadcast(&cv);
      //  std::cout << "hi" << std::endl;
    }
    pthread_mutex_unlock(&mutex);
}
