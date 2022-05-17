
<div align="center">

# Reinforcement learning models of decision making

</div>

This repository contains materials for modeling choice data recorded while participants played a two-armed bandit task. Three reinforcement learning (RL) models are implemented along with two heuristic models for comparison. RL models are all variations of temporal difference learning models. 

These models were coded by Kaustubh Kulkarni in the spring of 2022 (reach him via <a target="_blank" rel="noopener noreferrer" href="mailto:kaustubh(dot)kulkarni(at)icahn(dot)mssm(dot)edu">email</a>, <a target="_blank" rel="noopener noreferrer" href="https://github.com/kulkarnik">GitHub</a>, or <a target="_blank" rel="noopener noreferrer" href="https://twitter.com/krkulkarni_">Twitter</a> with any questions).

## 🧮 Models
| **Model** |           **Model Description**          |       **Parameters**       |                                                         **Details**                                                        |
|:---------:|:----------------------------------------:|:--------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
|   Biased  | Preference for one machine               |            bias            | The bias parameter fits a preference for one machine over the other.                                                       |
| Heuristic | Switch to other machine after two losses |           epsilon          | The epsilon parameter fits random choice that does not adhere to the switching strategy.                                   |
|     RW    | Temporal difference learning (TDL)       |         alpha, beta        | Alpha refers to the learning rate, beta refers to the inverse temperature parameter.                                       |
|  RWDecay  | TDL with center decay                    |     alpha, decay, beta     | Alpha and beta are the same as above. The decay parameter fits the speed by which the values move towards a neutral value. |
|    RWRL   | TDL with separate learning rates         | alpha_pos, alpha_neg, beta | Alpha_pos and alpha_neg refer to learning rates for positive and negative prediction errors respectively.                  |

## 🙏 Acknowledgments
I am grateful to Dr. Xiaosi Gu, Dr. Daniela Schiller, and the Center for Computational Psychiatry at Mount Sinai. I am also grateful to Project Jupyter for making it possible to create and share these materials in a Jupyter Book.

## 🎫 License
<a rel="license" target="_blank" rel="noopener noreferrer" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />
Content in this repository (i.e., any .md or .ipynb files in the content/ folder) is licensed under a <a rel="license" target="_blank" rel="noopener noreferrer" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
