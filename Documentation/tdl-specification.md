# Texture Definition Language

*Texture Definition Language*, or *TDL* in short, is a scripting language in *SuperTerrain+* introduced in [v0.8.9](https://github.com/stephen-hqxu/superterrainplus/releases/tag/v0.8.9) which allows defining rules in a human-readable format for rule-based biome-dependent terrain texture splatting system.

Here is an example:

```css

#texture [redrock, grass, mossrock, sand, soil, darkrock]

#group view {
	redrock, mossrock, darkrock, sand, soil := (40u, 20u, 10u),
	grass := (64u, 60u, 55u)
}

#rule altitude {
	0u := (1.0f -> soil),
	1u := (0.5f -> sand, 0.72f -> mossrock, 1.0 -> grass)
}

#rule gradient {
	0u := (0.0f, 0.2f, 0.0f, 0.27f -> sand),
	1u := (0.4f, 0.65f, 0.58f, 0.8f -> darkrock, 0.625f, 1.0f, 0.65f, 1.0f -> redrock)
}

```

Similar to most programming language, *TDL* is sensitive to case and insensitive to white space and indentation.

# Directive

A directive is a controlling command. A directive always begins with a dash symbol `#`, followed by a directory name. Currently *TDL* supports the following type of directive:

- texture
- group
- rule

## Texture

A texture directive allows user to declare texture variables. A texture variable is consist of pure alphabets, no other symbols are allowed. The scope of declared texture variables start from the time they are declared until the end of the file. Variables should be declared in squared bracket `[]`.

> From [v0.15.10](https://github.com/stephen-hqxu/superterrainplus/releases/tag/v0.15.10), texture directive no longer ends with `;`.

More than one texture variable can be declared in the same texture directive. To do this, separate each variable by a comma `,`.

Texture variable only acts as a placeholder to be referenced in the engine in runtime and does not contains any value in the script. Hence, redeclaration of the same texture variable, either in the same texture directive block, or any other, has no effect.

An example shows a declaration of two texture variables, `snow` and `dirt`:

```markdown

#texture [snow, dirt]

```

This is completely equivalent to the following codes:

```markdown

#texture [snow]
#texture [dirt]

OR

#texture [snow, snow, dirt]

```

## Rule

A rule directive allows user to define texture splatting rule. Currently the engine supports the following rules:

- altitude
- gradient

The type of rule is followed by the name of directive, and after the specification of rule type it is then followed by a curly bracket block `{}` where rules are defined.

Each set of rules is defined per biome basis, and more than one rule can be defined for a biome. The rule set is begun with a biome ID, followed by a logical assignment operator `:=`, then the rule contents.

```cpp

   0u := (...)
// ^     -----
// biome  rule
//  ID   content

```

> You don't need the numeric suffix for the biome ID, I prefer typinig it because in the engine biome ID is defined as an unsigned integer type.

The rule contents are enclosed by a round bracket `()`, each rule content contains rule argument and active texture region, and an right arrow `->` is used to separate between these two.

```cpp

    0u := (0.3f -> rock)
//  ^      ----    ----
//biome	   rule    texture
// ID    argument   name

```

It is an error, if the texture name is not being declared by a texture directive previously.

> The actual semantics of the rule argument is specified by the user, because the biomemap and texture splatmap generators are both user programmable.

Both rule set and rule content are repeatable, just place a comma `,` between each of them.

```cpp

0u := (0.3f -> dirt, 1.0f -> rock),
1u := (0.1 -> sand)

```

### Altitude

An altitude rule is specified by a single real number in the rule argument as the upper bound below which a texture region should be made activated.

```css

#rule altitude {
	0u := (0.3f -> dirt, 1.0f -> snow),
	1u := (1.0f -> dirt)
}

```

### Gradient

A gradient rule is specified by 4 real numbers in the rule argument, being minimum gradient, maximum gradient, altitude lower bound and altitude upper bound, respectively.

```css

#rule gradient {
	0u := (0.1f, 0.45f, 0.1f, 0.5f -> snow, 
		0.5f, 0.9f, 0.1f, 0.45f -> dirt)
}

```

## Group

> Introduced in [v0.12.0](https://github.com/stephen-hqxu/superterrainplus/releases/tag/v0.12.0)

A group directive defines a texture group. A texture group contains a collection of texture which shares the same property. Currently the engine supports the following types of group:

- view

Similar to rule directive, the type of group follows the directive name, and the group is defined in a curly bracket-enclosed block.

### View

A view group defines the texture coordinate scale of all texture in a group.

```css

#group view {
	grass := (8u, 5u, 2u)
}

```

The engines currently supports adaptive triple scaling, meaning each view group allows specification of three scales, being primary, secondary and tertiary scale respectively. These scales are chosen by the renderer automatically in the shader.

Each group definition begins with a texture name which should have been declared in the current scope, followed by the logical assignment `:=`, and a round bracket enclosed group argument.

To declare multiple texture names, separate each of them by a comma `,`:

```css

grass, snow, soil := (12u, 8u, 1u)

```

Like in the texture directive, re-specification of the same texture in the same group will be silently ignored. Hence, the group above is exactly the same as:

```css

grass, grass, snow, soil := (12u, 8u, 1u)

```

Similarly, multiple groups can be defined in the same block, remember to separate them by a comma `,`:

```css

rock := (20u, 10u, 5u),
grass, soil, sand, snow := (64u, 60u, 55u)

```

However, re-specification of the same texture in different groups will have different effect. Texture view specification comes later overwrite the former. Therefore a equivalent script as the previous.

```css

rock, sand := (3u, 2u, 1u),
/* `rock` and `sand` texture are both reassigned by the groups after */
rock := (20u, 10u, 5u),
grass, soil, sand, snow := (64u, 60u, 55u)

```

# Usage

A *TDL* script can be parsed by `SuperTerrainPlus::STPAlgorithm::STPTextureDefinitionLanguage`, and the data defined in a *TDL* script can then be loaded into a texture database.

Remember, texture names declared earlier have no real meaning other than being reference values. Therefore the parser will convert texture names into some IDs pointing into the texture database being added, and then user can manipulate the database further using this piece of information, for example uploading texture map data.

In case there is a syntax or semantics error during this process, *TDL* lexer and parser will throw an exception containing reasons for the error at which character in which line.