#pragma once
#ifndef _STP_OBJECT_POOL_H_
#define _STP_OBJECT_POOL_H_

//System
#include <queue>
#include <mutex>

namespace SuperTerrainPlus {

	/**
	 * @brief STPObjectPool is a pool for reusing objects that are expensive to re-construct in a multithreaded environment.
	 * @tparam T Denotes the type of the object in the pool. It must be movable.
	 * @tparam New Provide a creator of object T when the pool runs out of available objects at request.
	 * Call to the creator is atomic.
	*/
	template<class T, class New>
	class STPObjectPool {
	private:

		New Creator;
		std::queue<T> ObjectPool;

		std::mutex PoolLock;

	public:

		/**
		 * @brief Initialise an object pool.
		 * @tparam ...Arg The argument type to initialise the creator.
		 * @param creator_arg... Arguments list to construct the object creator. 
		*/
		template<typename... Arg>
		STPObjectPool(Arg&&...);

		STPObjectPool(const STPObjectPool&) = delete;

		STPObjectPool(STPObjectPool&&) = delete;

		STPObjectPool& operator=(const STPObjectPool&) = delete;

		STPObjectPool& operator=(STPObjectPool&&) = delete;

		~STPObjectPool() = default;

		/**
		 * @brief Request an object from the current object pool.
		 * If the object pool is empty, create a new object and return.
		 * @tparam ...Arg The argument type to invoke the creator, when there is no object available.
		 * @param creator_arg... Arguments list to invoke the functor of creator when creation of object is demanded.
		 * @return The requesting object.
		*/
		template<typename... Arg>
		T requestObject(Arg&&...);

		/**
		 * @brief Return the object into the object pool.
		 * It is an undefined behaviour if the object is not requested from the pool.
		 * @param obj The object to be returned.
		*/
		void returnObject(T&&);

	};
}
#include "STPObjectPool.inl"
#endif//_STP_OBJECT_POOL_H_