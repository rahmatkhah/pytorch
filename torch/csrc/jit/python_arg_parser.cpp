#include "python_arg_parser.h"

namespace torch { namespace jit { namespace python {

struct ParsedArgs {
  std::vector<py::handle> vars;
  std::string desc;
  bool is_volatile = false;
};

namespace {

template<typename T>
py::object cast_handle_sequence(std::vector<py::handle> objs) {
  auto num_objs = objs.size();
  T sequence { num_objs };
  for (std::size_t i = 0; i < num_objs; ++i)
    sequence[i] = py::reinterpret_borrow<py::object>(objs[i]);
  return sequence;
}

void flatten_rec(PyObject* obj, ParsedArgs& args) {
  if (PyTuple_Check(obj)) {
    args.desc.push_back('(');
    for (auto item : py::reinterpret_borrow<py::tuple>(obj))
      flatten_rec(item.ptr(), args);
    args.desc.push_back(')');
  } else if (PyList_Check(obj)) {
    args.desc.push_back('[');
    for (auto item : py::reinterpret_borrow<py::list>(obj))
      flatten_rec(item.ptr(), args);
    args.desc.push_back(']');
  } else if (THPVariable_Check(obj)) {
    auto& var = reinterpret_cast<THPVariable*>(obj)->cdata;
    args.vars.push_back(obj);
    args.is_volatile |= var.is_volatile();
    if (args.is_volatile) {
      args.desc.push_back('v');
    } else {
      args.desc.push_back(var.requires_grad() ? 'r' : 'n');
    }
  } else {
    std::string msg = "Only tuples, lists and Variables supported as JIT inputs, but got ";
    msg += THPUtils_typename(obj);
    throw std::runtime_error(msg);
  }
}

void mark_all_volatile(std::string& desc) {
  auto desc_size = desc.size();
  for (std::size_t i = 0; i < desc_size; ++i) {
    if (desc[i] == 'r' || desc[i] == 'n')
      desc[i] = 'v';
    else if (desc[i] == 'v')
      break;
  }
}

} // anonymous namespace

flattened_args flatten(py::handle obj) {
  ParsedArgs args;
  flatten_rec(obj.ptr(), args);
  // We might have put some Variable descriptors in desc before we discovered
  // the first volatile one, so we need to fix it now.
  if (args.is_volatile) {
    mark_all_volatile(args.desc);
  }
  return std::make_tuple(cast_handle_sequence<py::tuple>(args.vars), py::bytes(args.desc), args.is_volatile);
}

namespace {

using tuple_iterator = decltype(std::declval<py::tuple>().begin());

template<typename T>
py::object cast_sequence(std::vector<py::object> objs) {
  auto num_objs = objs.size();
  T sequence { num_objs };
  for (std::size_t i = 0; i < num_objs; ++i)
    sequence[i] = std::move(objs[i]);
  return sequence;
}

py::object unflatten_rec(tuple_iterator& var_it,
                         tuple_iterator& var_it_end,
                         std::string::iterator& desc_it) {
  char type = *desc_it++;
  if (type == '(') {
    std::vector<py::object> objs;
    while (*desc_it != ')')
      objs.push_back(unflatten_rec(var_it, var_it_end, desc_it));
    ++desc_it;
    return cast_sequence<py::tuple>(objs);
  } else if (type == '[') {
    std::vector<py::object> objs;
    while (*desc_it != ']')
      objs.push_back(unflatten_rec(var_it, var_it_end, desc_it));
    ++desc_it;
    return cast_sequence<py::list>(objs);
  } else {
    if (var_it == var_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto var = *var_it++;
    return py::reinterpret_borrow<py::object>(var);
  }
}

} // anonymous namespace

py::object unflatten(py::tuple vars, py::bytes descriptor) {
  std::string desc = descriptor; // <sigh> we have to make a copy
  auto vars_it = vars.begin();
  auto vars_it_end = vars.end();
  auto desc_it = desc.begin();
  return unflatten_rec(vars_it, vars_it_end, desc_it);
}

}}} // namespace torch::jit::python
