#ifdef _STP_OBJECT_POOL_H_

template<class T, class New>
template<typename... Arg>
inline SuperTerrainPlus::STPObjectPool<T, New>::STPObjectPool(Arg&&... creator_arg) :
	Creator(std::forward<Arg>(creator_arg)...) {

}

template<class T, class New>
template<typename... Arg>
inline T SuperTerrainPlus::STPObjectPool<T, New>::request(Arg&&... creator_arg) {
	const std::unique_lock request_lock(this->PoolLock);

	if (this->ObjectPool.empty()) {
		//no more object available, request a new one and return
		return this->Creator(std::forward<Arg>(creator_arg)...);
	}
	//there are available object, dequeue from the queue and return
	T object(std::move(this->ObjectPool.front()));
	this->ObjectPool.pop();
	return object;
}

template<class T, class New>
inline void SuperTerrainPlus::STPObjectPool<T, New>::release(T&& obj) {
	const std::unique_lock return_lock(this->PoolLock);

	//return object back to the queue
	this->ObjectPool.emplace(std::move(obj));
}

#endif//_STP_OBJECT_POOL_H_