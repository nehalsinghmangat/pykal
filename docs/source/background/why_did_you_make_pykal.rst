=======================
Why did you make pykal?
=======================

Good question. After all, I am not an engineer. My background is in physics and mathematics; I hold a bachelors in the former and a masters in the latter. To wit: if you held a gun to my head and told me to design a paper bag, I would tell my family I love them and then tell you to pull the trigger. So it is strange that someone like me would make a robotics framework; and truthfully, I wasn't drawn to robotics because of the money, or the status, or the hordes of women who would fall at my feet. Rather, I was drawn to the field by its progenitor: control theory. 

Control theory is at once elegant and powerful. If control theory is fundamental to all fields of engineering, then it is the field entire of robotics. I loved reading about estimation algorithms using Lie theory, or modeling drones using differential geometry, but I found, to my dismay, that there was no easy way to implement these ideas. Theory is beautiful, yes, but practice is validating, so why were so many promising theories never put onto hardware? 

After working in academia for some time, I've come up with a theory of my own to explain this phenomenon: it's hard. That is, the journey from head to hardware is long, messy, and often leaves in its wake a horrifying tangle of code and third-party software that no one but the author (and perhaps his research group) can use. And maybe not this time, but the next time, or the next time, or the time after that, when the original author has left or an outsider tries to follow his path, something will break and that's that. 

This is a tragedy. This is an unacceptable state of affairs. This is why I made **pykal**, a portmanteau of "**py**thon" and "I can't believe it's this hard to implement a **kal**man filter in ROS". Like the child of a failing marriage, this framework was born out of frustration, and the unhappy coupling it sought to solve was that of theoretical control systems and practical robotics. But unlike the child of a failing marriage, I firmly believe **pykal** can fix things.

Let's get started.

----

:doc:`← Overview of pykal <overview_of_pykal>` | :doc:`pykal_core tutorials → <why_did_you_make_pykal>`

----   

