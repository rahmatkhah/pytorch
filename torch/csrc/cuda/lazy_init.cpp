#include "lazy_init.h"

#include <Python.h>
#include <mutex>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"

namespace torch {
namespace cuda {

void lazy_init() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto module = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
    if (!module) throw python_error();
    auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) throw python_error();
  });
}

}
}
