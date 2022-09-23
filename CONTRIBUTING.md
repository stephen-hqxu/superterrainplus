# :seedling: Contributing to SuperTerrain+

Contributions are what make the open source community such an amazing place to learn, inspire, and create. I am absolutely glad to hear that you want to contribute something to *SuperTerrain+*. As a new project, there are still a significant number of places to be improved and worked on. All contributions made by everyone, from simple ones like fixing a typo in the documentation, to big improvement such as bug fix and new feature, are absolutely welcome and very much appreciated.

This document aims to provide you a helpful guideline for contributing to the project.

## Commit

If you have a brilliant idea, please fork this repository and create a pull request. The process is exactly the same as how most open source repositories are done.

1. Fork the repository.
1. Create and push a new branch to your fork.
1. Start your creation. Don't forget to push your commit to the new branch in your fork.
1. Open a pull request to merge into the master branch in the root repository.

When you are committing changes, briefly describe what has been achieved in this commit in the commit message. You might notice that I tend to prefix *emoji* before the message, and you don't have to repeat my approach, it just makes myself identifying what has been changed by the same commit easier; although you are absolutely welcome if you wish to.

As always, you are welcome to open an issue if you found something that does not work, or simply have any question.

## Code style

A consistent code style improves readability of codebase and helps future developers getting start with the development quickly. To make formatting code simpler for our contributors, *SuperTerrain+* provides a custom `.clang-format` file located at the project root. Most modern IDEs are integrated with clang formatter to help you maintaining the style as you type.

However, currently clang-format is still imperfect and cannot fully replicate my complex coding style, so please do not use it to format the entire file, and only do so on the newly changed code; this also helps reducing the size of diffs and makes code review easier.

### Naming convention

- All files, namespaces, structures and classes should have their names prefixed with *STP*, followed by a Pascal-style name.
- Name of member variables should follow the same naming convention as above but without the *STP* prefix.
- Name of functions should use Camel-style.
- Name of local variable does not matter, but it is recommended to use either Camel-style or Snake-style.
- All member functions and member variables from within the class should always be called with `this->`. Hence no prefix of *m* or something similar should be added.
- Include guards should be given a name pattern of `_{FILENAME}_{EXTENSION}_` and capitalised everything. Some examples are:

```cpp

//STPRenderer.h
#ifndef _STP_RENDERER_H_

//STPTerrainRenderer.hpp
#ifndef _STP_TERRAIN_RENDERER_HPP_

```

### Conditional statement

- When working with `if`, consider if you can reduce the amount of code enclosed inside the block by flipping the logic over, and avoid using `else` when it is unnecessary. For example:

```cpp

//Bad
if (!hasError) {
	/* 1k lines of code inside the `if` block */
	return value;
	
} else { //unnecessary `else` after return
	throw std::exception("Something went wrong!");
}

//Good
if(hasError) {
	throw std::exception("Something went wrong!");
}

/* 1k lines of code */
return value;

```

- When working with `switch`, avoid using `break` when it is unnecessary, such as using `return` in all `case`s.
- Always wrap the body with brace, even when there is only one line.

### Variable usable

- I can safely claim that over 90 precents of integer variables used in this project are unsigned, hence avoid using a signed type if it is not necessary, to avoid superfluous bugs due to overlooked integer promotion. Occasionally you may want to represent something like *an invalid state* with -1, which should be replaced by a max value of the corresponding unsigned type like `std::numeric_limits<unsigned int>::max()`.
- Always suffix a numeric variable based on their type, like *123u* for unsigned integer and *3.14f* for float. This is also to avoid unintentional integer promotion and generation of double-precision instruction.
- If applicable, always declare a member function and local variable as `const`; for member variable, do it strategically to allow effective use of copy assignment.

---