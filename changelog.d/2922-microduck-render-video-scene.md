### Added: the Microduck render example can reach a variant scene

`docs/policies/microduck.md` carries the skill-to-scene table, and four of the
nine shipped Pollen weights need a scene the registry entry does not declare.
`changelog.d/2900-microduck-skill-scenes.md` said so when it corrected the page:
"`examples/microduck/render_video.py` builds exactly that, which is why the page
said any shipped weight 'drops straight in'". The page was corrected; the example
it names was not. It built `Robot("microduck", mesh=False)` and had no flag for
anything else, so the by-path route the page documents in prose was reachable
from no shipped example.

`--scene` names a scene file under the Microduck asset directory. The three
scenes that ship in the one asset directory the entry already downloads:

    scene.xml          njnt=15  nu=14  nq=21  wheel joints=0  ball bodies=0
    scene_rollers.xml  njnt=19  nu=14  nq=25  wheel joints=4  ball bodies=0
    scene_ball.xml     njnt=16  nu=14  nq=28  wheel joints=0  ball bodies=1

`roller` and `roller_crouch` need the four wheels only `scene_rollers.xml`
carries, and the `ball_kick` pair needs the prop only `scene_ball.xml` places.
`nu` is 14 in all three, which is why nothing refuses the mismatch: the policy
writes exactly the same fourteen control targets whichever scene is loaded, and
the physics has nothing to roll on or nothing to kick. The run reports success
and the rendered video shows a duck going nowhere with no indication why.

The scene is resolved by name through `strands_robots.utils.get_search_paths` -
the same route the page documents - rather than as a path the caller spells out,
so all three are reachable by the names the table uses. A name no asset root
carries is refused, naming the directories that were searched: a misspelled
`--scene` that quietly resolved the entry's declared scene would render that same
duck-going-nowhere and report success, which is the failure the flag exists to
remove.

Omitting `--scene` forwards no asset at all, so the five weights that run on the
declared scene resolve the registry entry by name exactly as before. No library
code changes.
