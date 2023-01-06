/**
* @file
* @brief  Context Manager
*/
#ifndef TOTRI_COMMON_CONTEXT_MANAGER_H
#define TOTRI_COMMON_CONTEXT_MANAGER_H

#include <map>
#include <torch/extension.h>

namespace totri {

template<typename T>
class ContextManager
{
 public:
	static T* Get(int device) {
    std::map<int, T*>& contexts = GetContexts();
    auto it = contexts.find(device);
    if (it != contexts.end()) {
      return it->second;
    }
    T* new_context = new T(device);
    contexts[device] = new_context;
    return new_context;
  }

  static void Free(int device) {
    std::map<int, T*>& contexts = GetContexts();
    auto it = contexts.find(device);
    if (it != contexts.end()) {
      delete (*it).second;
      contexts.erase(it);
    }
  }

  static void FreeAll() {
    std::map<int, T*>& contexts = GetContexts();
    for (auto it = contexts.begin(); it != contexts.end(); ++it) {
      delete (*it).second;
    }
    contexts.clear();
  }

 private:
	ContextManager();
  ~ContextManager();
	ContextManager(const ContextManager<T>&) = delete;
	ContextManager<T>& operator= (const ContextManager<T>&) = delete;
  static std::map<int, T*>& GetContexts() {
    static std::map<int, T*> contexts;
    return contexts;
  }
};

} // namespace totri

#endif // TOTRI_COMMON_CONTEXT_MANAGER_H
