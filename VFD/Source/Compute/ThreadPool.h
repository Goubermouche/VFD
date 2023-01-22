#ifndef THREAD_POOL
#define THREAD_POOL

#include "pch.h"

namespace vfd
{
    using concurencyT = std::invoke_result_t<decltype(std::thread::hardware_concurrency)>;

    /// <summary>
    /// A helper class to facilitate waiting for and/or getting the results of multiple futures at once.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    template <typename T>
    class MultiFuture
    {
    public:
        /// <summary>
        /// Construct a multi_future object with the given number of futures.
        /// </summary>
        /// <param name="num_futures_">The desired number of futures to store.</param>
        explicit MultiFuture(const size_t num_futures_ = 0) : futures(num_futures_) {}

        /// <summary>
        /// Get the results from all the futures stored in this multi_future object.
        /// </summary>
        /// <returns>A vector containing the results.</returns>
        std::vector<T> Get()
        {
            std::vector<T> results(futures.size());
            for (size_t i = 0; i < futures.size(); ++i) {
                results[i] = futures[i].get();
            }

            return results;
        }

        /// <summary>
        /// Wait for all the futures stored in this multi_future object.
        /// </summary>
        void Wait() const
        {
            for (size_t i = 0; i < futures.size(); ++i) {
                futures[i].wait();
            }
        }

        std::vector<std::future<T>> futures;
    };

    /// <summary>
    /// A fast, lightweight, and easy-to-use C++17 thread pool class.
    /// </summary>
    class ThreadPool : public RefCounted
    {
    public:
        explicit ThreadPool(const concurencyT threadCount = std::thread::hardware_concurrency()) : m_ThreadCount(threadCount ? threadCount : std::thread::hardware_concurrency()), m_Threads(std::make_unique<std::thread[]>(threadCount ? threadCount : std::thread::hardware_concurrency()))
        {
            CreateThreads();
        }

        ~ThreadPool()
        {
            // wait for all tasks to finish
            WaitForTasks();
            DestroyThreads();
        }

        /// <summary>
        ///  Get the number of tasks currently waiting in the queue to be executed by the threads.
        /// </summary>
        /// <returns>The number of queued tasks.</returns>
        size_t GetTasksQueued() const
        {
            const std::scoped_lock tasks_lock(m_TasksMutex);
            return m_Tasks.size();
        }

        /// <summary>
        /// Get the number of tasks currently being executed by the threads.
        /// </summary>
        /// <returns>The number of running tasks.</returns>
        size_t GetTasksRunning() const
        {
            const std::scoped_lock tasks_lock(m_TasksMutex);
            return m_TasksTotal - m_Tasks.size();
        }

        /// <summary>
        /// Get the total number of unfinished tasks: either still in the queue, or running in a thread. Note that get_tasks_total() == get_tasks_queued() + get_tasks_running().
        /// </summary>
        /// <returns>The total number of tasks.</returns>
        size_t GetTasksTotal() const
        {
            return m_TasksTotal;
        }

        /// <summary>
        /// Get the number of threads in the pool.
        /// </summary>
        /// <returns> The number of threads.</returns>
        concurencyT GetThreadCount() const
        {
            return m_ThreadCount;
        }

        /// <summary>
        /// Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue.
        /// </summary>
        /// <typeparam name="F">The type of the function to loop through.</typeparam>
        /// <typeparam name="T1">The type of the first index in the loop. Should be a signed or unsigned integer.</typeparam>
        /// <typeparam name="T2">The type of the index after the last index in the loop. Should be a signed or unsigned integer. If T1 is not the same as T2, a common type will be automatically inferred.</typeparam>
        /// <typeparam name="T">The common type of T1 and T2.</typeparam>
        /// <typeparam name="R">The return value of the loop function F (can be void).</typeparam>
        /// <param name="firstIndex">The first index in the loop.</param>
        /// <param name="indexAfterLast">The index after the last index in the loop, the loop will iterate from first_index to (index_after_last - 1) inclusive.</param>
        /// <param name="loop"> The function to loop through - called once per block, should take exactly two arguments - the first index in the block and the index after the last index in the block.</param>
        /// <param name="blockCount">The maximum number of blocks to split the loop into. The default is to use the number of threads in the pool.</param>
        /// <returns>A multi_future object that can be used to wait for all the blocks to finish. If the loop function returns a value, the multi_future object can be used to obtain the values returned by each block.</returns>
        template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
        MultiFuture<R> ParallelizeLoop(const T1& firstIndex, const T2& indexAfterLast, const F& loop, size_t blockCount = 0)
        {
            T firstIndexT = static_cast<T>(firstIndex);
            T indexAfterLastT = static_cast<T>(indexAfterLast);
            if (firstIndexT == indexAfterLastT) {
                return MultiFuture<R>();
            }

            if (indexAfterLastT < firstIndexT) {
                std::swap(indexAfterLastT, firstIndexT);
            }

            if (blockCount == 0) {
                blockCount = m_ThreadCount;
            }

            const size_t totalSize = static_cast<size_t>(indexAfterLastT - firstIndexT);
            size_t blockSize = static_cast<size_t>(totalSize / blockCount);

            if (blockSize == 0)
            {
                blockSize = 1;
                blockCount = totalSize > 1 ? totalSize : 1;
            }

            MultiFuture<R> mf(blockCount);
            for (size_t i = 0; i < blockCount; ++i)
            {
                const T start = (static_cast<T>(i * blockSize) + firstIndexT);
                const T end = (i == blockCount - 1) ? indexAfterLastT : (static_cast<T>((i + 1) * blockSize) + firstIndexT);
                mf.futures[i] = Submit(loop, start, end);
            }

            return mf;
        }

        /// <summary>
        /// Push a function with zero or more arguments, but no return value, into the task queue.
        /// </summary>
        /// <typeparam name="F">The type of the function.</typeparam>
        /// <typeparam name="...A">The types of the arguments.</typeparam>
        /// <param name="task">The function to push.</param>
        /// <param name="...args">The arguments to pass to the function.</param>
        template <typename F, typename... A>
        void PushTask(const F& task, const A&... args)
        {
            {
                const std::scoped_lock tasks_lock(m_TasksMutex);
                if constexpr (sizeof...(args) == 0) {
                    m_Tasks.push(std::function<void()>(task));
                }
                else {
                    m_Tasks.push(std::function<void()>([task, args...]{ task(args...); }));
                }
            }
            ++m_TasksTotal;
            m_TaskAvailable.notify_one();
        }

        /// <summary>
        /// Reset the number of threads in the pool. Waits for all currently running tasks to be completed, then destroys all threads in the pool and creates a new thread pool with the new number of threads. Any tasks that were waiting in the queue before the pool was reset will then be executed by the new threads. If the pool was paused before resetting it, the new pool will be paused as well.
        /// </summary>
        /// <param name="newThreadCount">The number of threads to use. The default value is the total number of hardware threads available, as reported by the implementation. This is usually determined by the number of cores in the CPU. If a core is hyperthreaded, it will count as two threads.</param>
        void Reset(const concurencyT threadCount = std::thread::hardware_concurrency())
        {
            const bool was_paused = paused;
            paused = true;
            WaitForTasks();
            DestroyThreads();
            m_ThreadCount = threadCount ? threadCount : std::thread::hardware_concurrency();
            m_Threads = std::make_unique<std::thread[]>(m_ThreadCount);
            paused = was_paused;
            CreateThreads();
        }

        /// <summary>
        /// Submit a function with zero or more arguments into the task queue.  If the function has a return value, get a future for the eventual returned value. If the function has no return value, get an std::future void which can be used to wait until the task finishes.
        /// </summary>
        /// <typeparam name="F">The type of the function.</typeparam>
        /// <typeparam name="...A">The types of the zero or more arguments to pass to the function.</typeparam>
        /// <typeparam name="R">The return type of the function (can be void).</typeparam>
        /// <param name="task">The function to submit.</param>
        /// <param name="...args">The zero or more arguments to pass to the function.</param>
        /// <returns>A future to be used later to wait for the function to finish executing and/or obtain its returned value if it has one.</returns>
        template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
        std::future<R> Submit(const F& task, const A&... args)
        {
            std::shared_ptr<std::promise<R>> task_promise = std::make_shared<std::promise<R>>();
            PushTask([task, args..., task_promise]
                {
                    try
                    {
                        if constexpr (std::is_void_v<R>)
                        {
                            task(args...);
                            task_promise->set_value();
                        }
                        else
                        {
                            task_promise->set_value(task(args...));
                        }
                    }
                    catch (...)
                    {
                        try
                        {
                            task_promise->set_exception(std::current_exception());
                        }
                        catch (...)
                        {
                        }
                    }
                });
            return task_promise->get_future();
        }

        /// <summary>
        /// Wait for tasks to be completed. Normally, this function waits for all tasks, both those that are currently running in the threads and those that are still waiting in the queue. However, if the pool is paused, this function only waits for the currently running tasks (otherwise it would wait forever). Note: To wait for just one specific task, use submit() instead, and call the wait() member function of the generated future.
        /// </summary>
        void WaitForTasks()
        {
            m_Waiting = true;
            std::unique_lock<std::mutex> tasks_lock(m_TasksMutex);
            m_TaskDone.wait(tasks_lock, [this] {
                return (m_TasksTotal == (paused ? m_Tasks.size() : 0));
            });
            m_Waiting = false;
        }

        /// <summary>
        /// An atomic variable indicating whether the workers should pause. When set to true, the workers temporarily stop retrieving new tasks out of the queue, although any tasks already executed will keep running until they are finished. Set to false again to resume retrieving tasks.
        /// </summary>
        std::atomic<bool> paused = false;

    private:
        /// <summary>
        /// Create the threads in the pool and assign a worker to each thread.
        /// </summary>
        void CreateThreads()
        {
            m_Running = true;
            for (concurencyT i = 0; i < m_ThreadCount; ++i)
            {
                m_Threads[i] = std::thread(&ThreadPool::Worker, this);
            }
        }

        /// <summary>
        /// Destroy all threads in the pool.
        /// </summary>
        void DestroyThreads()
        {
            m_Running = false;
            m_TaskAvailable.notify_all();
            for (concurencyT i = 0; i < m_ThreadCount; ++i)
            {
                m_Threads[i].join();
            }
        }

        /// <summary>
        /// A worker function to be assigned to each thread in the pool. Waits until it is notified by push_task() that a task is available, and then retrieves the task from the queue and executes it. Once the task finishes, the worker notifies wait_for_tasks() in case it is waiting.
        /// </summary>
        void Worker()
        {
            while (m_Running)
            {
                std::function<void()> task;
                std::unique_lock<std::mutex> tasks_lock(m_TasksMutex);
                m_TaskAvailable.wait(tasks_lock, [&] {
                    return !m_Tasks.empty() || !m_Running;
                    });
                if (m_Running && !paused)
                {
                    task = std::move(m_Tasks.front());
                    m_Tasks.pop();
                    tasks_lock.unlock();
                    task();

                    --m_TasksTotal;
                    if (m_Waiting) {
                        m_TaskDone.notify_one();
                    }
                }
            }
        }

        /// <summary>
        /// An atomic variable indicating to the workers to keep running. When set to false, the workers permanently stop working.
        /// </summary>
        std::atomic<bool> m_Running = false;

        /// <summary>
        /// A condition variable used to notify worker() that a new task has become available.
        /// </summary>
        std::condition_variable m_TaskAvailable = {};

        /// <summary>
        /// A condition variable used to notify wait_for_tasks() that a tasks is done.
        /// </summary>
        std::condition_variable m_TaskDone = {};

        /// <summary>
        /// A queue of tasks to be executed by the threads.
        /// </summary>
        std::queue<std::function<void()>> m_Tasks = {};

        /// <summary>
        /// An atomic variable to keep track of the total number of unfinished tasks - either still in the queue, or running in a thread.
        /// </summary>
        std::atomic<size_t> m_TasksTotal = 0;

        /// <summary>
        /// A mutex to synchronize access to the task queue by different threads.
        /// </summary>
        mutable std::mutex m_TasksMutex = {};

        /// <summary>
        /// The number of threads in the pool.
        /// </summary>
        concurencyT m_ThreadCount = 0;

        /// <summary>
        /// A smart pointer to manage the memory allocated for the threads.
        /// </summary>
        std::unique_ptr<std::thread[]> m_Threads = nullptr;

        /// <summary>
        /// An atomic variable indicating that wait_for_tasks() is active and expects to be notified whenever a task is done.
        /// </summary>
        std::atomic<bool> m_Waiting = false;
    };
}

#endif //! THREAD_POOL