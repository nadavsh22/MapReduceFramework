//
// Created by nadavsh22 on 5/25/18.
//


#include <sys/types.h>
#include <atomic>
#include <pthread.h>
#include <cstdio>
#include <algorithm>
#include "MapReduceFramework.h"
#include "Barrier.h"
#include <semaphore.h>
#include <iostream>

class ThreadContext;


class ThreadContext
{
public:
    ThreadContext(const MapReduceClient &client, Barrier *bp, const int thread_id,
                  std::atomic<int> *atomic_counter, const InputVec *pInputVector,
                  OutputVec *pOutputVector,
                  const int numThreads,
                  sem_t *semaphore, std::vector<IntermediateVec *> *pQueue,
                  pthread_mutex_t *pQueueMutex,
                  bool *allWorkSplit, bool *shuffleStarted, pthread_mutex_t *pOutputVectorMutex,
                  std::vector<ThreadContext *> *pContextsVector)
            : client(client), bp(bp),
              thread_id(thread_id),
              atomic_counter(
                      atomic_counter),
              pInputVector(
                      pInputVector),
              pOutputVector(
                      pOutputVector),
              numThreads(numThreads),

              semaphore(semaphore),
              pQueue(pQueue),
              pQueueMutex(
                      pQueueMutex),
              allWorkSplit(
                      allWorkSplit),
              shuffleStarted(shuffleStarted),
              pOutputVectorMutex(
                      pOutputVectorMutex),
              intermediateVector(0),
              pContextsVector(pContextsVector)
    {
    }


    const MapReduceClient &client;
    Barrier *bp;
    const int thread_id;
    std::atomic<int> *atomic_counter;
    const InputVec *pInputVector;
    OutputVec *pOutputVector;
    const int numThreads;
    sem_t *semaphore;
    std::vector<IntermediateVec *> *pQueue;
    pthread_mutex_t *pQueueMutex;
    bool *allWorkSplit;
    bool *shuffleStarted;
    pthread_mutex_t *pOutputVectorMutex;
    IntermediateVec intermediateVector;
    std::vector<ThreadContext *> *pContextsVector; // pointers to the contexts array
};

/**
 * creates ThreadContexts
 * @param client
 * @param inputVec
 * @param outputVec
 * @param multiThreadLevel number of threads to create
 * @param atomic_counter
 * @param bp
 * @param workSplit boolean indicator for splitting work
 * @param shuffleStarted boolean indicator for shuffle
 * @param pOutputVectorMutex
 * @param queue_semaphore
 * @param pQueueMutex
 * @param queue
 * @param pContextsVector
 */
void createContexts(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec,
                    int multiThreadLevel, std::atomic<int> *atomic_counter,
                    Barrier *bp,
                    bool *workSplit, bool *shuffleStarted,
                    pthread_mutex_t *pOutputVectorMutex,
                    sem_t *queue_semaphore, pthread_mutex_t *pQueueMutex,
                    std::vector<IntermediateVec *> *queue,
                    std::vector<ThreadContext *> *pContextsVector)
{
    for (unsigned int i = 0; i < (unsigned) multiThreadLevel; ++i)
    {

        auto pNewContext = new ThreadContext(client, bp, i, atomic_counter, &inputVec,
                                             &outputVec, multiThreadLevel,
                                             queue_semaphore,
                                             queue, pQueueMutex, workSplit,
                                             shuffleStarted, pOutputVectorMutex, pContextsVector);

        pContextsVector->push_back(pNewContext);
    }
}

/**
 * deletes all contexts created
 * @param pContextsVector
 */
void deleteContexts(std::vector<ThreadContext *> *pContextsVector)
{
    for (auto &pThreadContext : *pContextsVector)
    {
        delete pThreadContext;
    }
}


//forward declaration for the shuffle function
void shuffle(int numThreads, std::vector<ThreadContext *> *pContextsVector);

void mapPhase(ThreadContext *tc);

void reducePhase(ThreadContext *tc);

bool equalTypes(K2 *left, K2 *right);


// ==============================
// =                            =
// =       THREAD ACTIONS       =
// =                            =
// ==============================
void
shufflesLittleHelper(std::vector<ThreadContext *> *pContextsVector, IntermediateVec &collector);

/**
 * a threads to do list
 * @param arg
 * @return
 */
void *threadActions(void *arg)
{
    auto tc = (ThreadContext *) arg;

    mapPhase(tc);

    // barrier
    tc->bp->barrier(); // waits for everyone to reach barrier

    if (tc->thread_id == 0)
    {
        shuffle(tc->numThreads, tc->pContextsVector);
    }

    reducePhase(tc);
    return nullptr;
}

// produces a (K2*, V2*) pair.
void emit2(K2 *key, V2 *value, void *context)
{

    IntermediatePair outputPair = IntermediatePair(key, value);
    auto tcp = (ThreadContext *) context;
    tcp->intermediateVector.push_back(outputPair);
}


// produces a (K3*, V3*) pair.
void emit3(K3 *key, V3 *value, void *context)
{
    OutputPair outputPair = OutputPair(key, value);
    auto tcp = (ThreadContext *) context;
    auto pOutputVector = tcp->pOutputVector;
    auto pOutVecMutex = tcp->pOutputVectorMutex;
    pthread_mutex_lock(pOutVecMutex);
    pOutputVector->push_back(outputPair);
    pthread_mutex_unlock(pOutVecMutex);
}

/**
 * comparator of pair keys
 * @param left
 * @param right
 * @return
 */
bool compareInterPairs(const IntermediatePair &left, const IntermediatePair &right)
{
    auto key1 = left.first;
    auto key2 = right.first;
    return *key1 < *key2;
}

// ==============================
// =                            =
// =            MAP             =
// =                            =
// ==============================
/**
 * runs the map phase
 * @param tc  ThreadContext
 */
void mapPhase(ThreadContext *tc)
{
    std::atomic<int> *atomicPointer = tc->atomic_counter;

    int old_value = ((*atomicPointer)++);
    auto inputSize = tc->pInputVector->size();
    while ((unsigned) old_value < inputSize)
    {
        auto pair = tc->pInputVector->at((unsigned) old_value);
        tc->client.map(pair.first, pair.second, tc);
        old_value = ((*atomicPointer)++);
    }
    std::sort(tc->intermediateVector.begin(), tc->intermediateVector.end(), compareInterPairs);
}


// ==============================
// =                            =
// =           SHUFFLE          =
// =                            =
// ==============================
/**
 * runs the shuffle phase
 * @param numThreads
 * @param pContextsVector
 */
void shuffle(int numThreads, std::vector<ThreadContext *> *pContextsVector)
{


    std::vector<IntermediateVec *> vectorOfVectors(0);

    ThreadContext *tcp;
    for (unsigned int j = 0; j < (unsigned)numThreads; ++j)
    {
        tcp = pContextsVector->at(j);
        if (!tcp->intermediateVector.empty())
        {
            vectorOfVectors.push_back(&tcp->intermediateVector);
        }
    }

    IntermediateVec collector;
    for (unsigned int k = 0; k < (unsigned)numThreads; ++k)
    {
        IntermediateVec *tempVector = &pContextsVector->at(k)->intermediateVector;
        for (IntermediatePair &pair: *tempVector)//shove all pairs to collector and sort it
        {
            collector.push_back(pair);
        }
    }

    std::sort(collector.begin(), collector.end(), compareInterPairs);

    shufflesLittleHelper(pContextsVector, collector); // now we divide the work between threads
    *pContextsVector->at(0)->allWorkSplit = true; // signify that shuffling ended
    for (unsigned int i = 0; i < (unsigned)numThreads; ++i)
    {
        sem_post(pContextsVector->at(0)->semaphore); // wake everyone up after work is allocated,
        // so no one is sleeping forever
    }
}
/**
 * divides work between the threads in pContextsVector
 * @param pContextsVector
 * @param collector sorted vector of intermediate pairs
 */
void
shufflesLittleHelper(std::vector<ThreadContext *> *pContextsVector, IntermediateVec &collector)
{
    K2 *curType = nullptr, *backType = nullptr;
    std::vector<IntermediatePair> *pSameKeyVector; // will hold vector of same key
    while (!collector.empty())// give all same key pairs to a specific thread to handle
    {
        pSameKeyVector = new std::vector<IntermediatePair>();

        IntermediatePair *backPair = &collector.back();

        curType = backPair->first;
        backType = backPair->first;

        while (equalTypes(backType, curType))
        {
            pSameKeyVector->push_back(*backPair);

            collector.pop_back();
            // now we look in the same thread's vector for additional elements of the same key
            if (collector.empty())
            { break; }
            backPair = &collector.back(); // next pair
            backType = backPair->first;
        }

        *pContextsVector->at(0)->shuffleStarted = true;
        // Push new vector (pSameKeyVector) to next thread
        pthread_mutex_lock(pContextsVector->at(0)->pQueueMutex);
        pContextsVector->at(0)->pQueue->push_back(pSameKeyVector);
        pthread_mutex_unlock(pContextsVector->at(0)->pQueueMutex);
        sem_post(pContextsVector->at(0)->semaphore);
    }
}

// ==============================
// =                            =
// =           REDUCE           =
// =                            =
// ==============================

/**
 * runs the reduce phase
 * @param tc ThreadContext
 */
void reducePhase(ThreadContext *tc)
{
    unsigned long size = 0;

    auto tmpMutex = tc->pQueueMutex;
    pthread_mutex_lock(tmpMutex);
    size = tc->pQueue->size();
    pthread_mutex_unlock(tmpMutex);


    while (!*tc->allWorkSplit || size > 0)
    {
        sem_wait(tc->semaphore); // decreases the semaphore and waits for it to become positive
        // wait for work to be added to queue
        pthread_mutex_lock(tmpMutex);
        auto queuePointer = tc->pQueue;
        size = queuePointer->size();
        if (!queuePointer->empty())
        {
            IntermediateVec *nextToReduce = tc->pQueue->back();
            queuePointer->pop_back();

            pthread_mutex_unlock(tmpMutex);

            tc->client.reduce(nextToReduce, tc);
            delete nextToReduce;
        }
        else
        {
            pthread_mutex_unlock(tmpMutex);
        }
    }
    for (unsigned int i = 0; i < (unsigned) tc->numThreads; ++i)
    {
        sem_post(tc->semaphore);
    }

}


// check whether two types are equal
bool equalTypes(K2 *left, K2 *right)
{
    bool retval = (!(*left < *right || *right < *left));
    return retval;

}

/**
 * The conductor of this orchestra
 * @param client
 * @param inputVec
 * @param outputVec
 * @param multiThreadLevel number of threads to create
 */
void runMapReduceFramework(const MapReduceClient &client,
                           const InputVec &inputVec, OutputVec &outputVec,
                           int multiThreadLevel)
{
    std::vector<ThreadContext *> *pContextsVector; // pointers to the contexts array
    pthread_t threads[multiThreadLevel - 1];

    auto pAtomCounter = new std::atomic<int>(0);

    Barrier barrier(multiThreadLevel);
    sem_t queue_semaphore;
    sem_init(&queue_semaphore, 0, 0);
    pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;

    bool workSplit = false, shuffleStarted = false; // used to signify when shuffle starts/ends;
    pthread_mutex_t output_vec_mutex = PTHREAD_MUTEX_INITIALIZER;
    pContextsVector = new std::vector<ThreadContext *>;
    std::vector<IntermediateVec *> queue(0);
    // create context for each thread
    createContexts(client, inputVec, outputVec, multiThreadLevel, pAtomCounter, &barrier,
                   &workSplit, &shuffleStarted, &output_vec_mutex, &queue_semaphore, &queue_mutex,
                   &queue, pContextsVector);

    for (unsigned int i = 0; i < (unsigned) (multiThreadLevel - 1); ++i)
    {
        pthread_create(&threads[i], nullptr, threadActions, pContextsVector->at(i + 1));
    }

    threadActions(pContextsVector->at(0)); // main thread runs with context(0)
    for (int i = 0; i < multiThreadLevel - 1; ++i)
    {
        pthread_join(threads[i], nullptr);
    }
    deleteContexts(pContextsVector);

    delete pContextsVector;
    delete pAtomCounter;
}