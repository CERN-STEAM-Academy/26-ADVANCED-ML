"""An embedded corpus of general English prose.

Why this module exists
----------------------
Notebook 2 measures *catastrophic forgetting*: the model is fine-tuned with RL on a
narrow arithmetic task, and we watch what happens to everything else it used to be able
to do. "Everything else" needs a concrete, fixed probe, and the probe must be:

* **Embedded.** Student VMs have unverified network access, so nothing may be downloaded
  at session time.
* **Redistributable.** The passages below are original prose written for this course and
  are covered by the repository licence. No corpus scraping, no licence archaeology.
* **Far from the training distribution.** The RL task is integer multiplication with
  answers in tags. This corpus is descriptive and narrative prose about coastlines,
  navigation and glass. It contains no digits and no arithmetic at all, so a perplexity
  rise on it cannot be explained away as "the model learned the task format".
* **Small.** Roughly two to three thousand tokens, which is one cheap forward pass. That
  matters because ``eval_general_perplexity`` is called every few steps *during*
  training, not just at the end.

The corpus is deliberately split into passages, each short enough to sit inside a modest
context window on a single forward pass. Note that ``eval_general_perplexity`` *pools* the
negative log-likelihood over the whole corpus rather than averaging per-passage
perplexities, so a longer passage does count for more - which is what "perplexity of the
corpus" conventionally means. Passage lengths here range from about 166 to 315 tokens, so
the two definitions would genuinely differ.
"""

from __future__ import annotations

PASSAGES: list[str] = [
    # --- Encyclopaedic register: physical geography -------------------------------
    """A sea cliff is not a wall so much as a slow argument between rock and water.
Waves arrive with a certain energy, and that energy has to go somewhere. Some of it is
spent on the simple mechanical shock of water striking stone. Some of it is spent
compressing air into joints and bedding planes, so that the rock is prised apart from
within rather than worn away from without. The rest is carried by the sand and shingle
the wave itself is holding, which acts as an abrasive and grinds the base of the cliff
into a notch. The notch deepens, the overhang above it grows heavier, and one winter the
overhang falls. What is left is a fresh vertical face, and the argument begins again a
little further inland.

The rate at which this happens depends less on the violence of individual storms than on
the character of the rock. Chalk and soft shale retreat quickly and produce cliffs that
are almost geometrically clean, because the whole face fails at once. Granite and hard
sandstone retreat slowly and unevenly, because failure follows the joints, and joints are
never regularly spaced. Between the two extremes lies most of the world's coastline, where
resistant beds alternate with weak ones and the sea exploits the difference. Headlands
form where the resistant beds run out to meet the water; bays form where the weak beds do.
Given time the headlands are cut back and the bays are filled, and a coast that began
ragged tends towards something smoother.""",
    """Beaches are best understood as reservoirs rather than surfaces. Sediment arrives from
rivers, from eroding cliffs, and from the sea floor offshore; it departs into deep water,
into dunes behind the beach, and along the shore under the influence of waves that arrive
at an angle. When arrivals and departures are in balance the beach appears permanent, and
people build on the assumption that it is. When the balance is disturbed the beach moves,
sometimes within a single season.

Longshore drift is the mechanism most often invoked and most often misunderstood. A wave
approaching obliquely runs up the beach along its own direction of travel but drains back
down the slope of the beach, which is to say straight out to sea. Each grain it carries
therefore returns a little to one side of where it started. Repeat this through a tide and
the whole beach is a conveyor belt. Interrupt the belt with a groyne or a harbour wall and
sediment piles up on the upstream side, which the builders regard as success, while the
beach on the downstream side starves, which the neighbours regard as theft. Coastal
engineering has produced a long literature on this problem and very few solutions that do
not simply move it somewhere else.""",
    """Estuaries occupy the awkward ground between river and sea, and they behave like
neither. Fresh water is lighter than salt water, so it tends to ride over the top of the
incoming tide rather than mixing with it, and the boundary between the two can be sharp
enough to see. Fine particles carried down by the river remain suspended in fresh water
but flocculate on contact with salt, clumping into aggregates heavy enough to sink. The
result is that estuaries trap sediment with remarkable efficiency, and left alone they
silt up and become marsh, then meadow, then dry land with a curious flatness to it.

The ecological consequence is out of all proportion to the area involved. Mudflats look
barren at low tide, but the mud is dense with worms, molluscs and crustaceans feeding on
organic matter the river has delivered, and the birds know it. Migratory waders time their
journeys around a handful of estuaries where they can refuel, which means that draining
one such site has effects thousands of miles away. This is the argument that has, slowly
and incompletely, shifted policy from reclamation towards protection.""",
    # --- Encyclopaedic register: technology and its history -----------------------
    """Before the invention of reliable timekeeping at sea, a navigator could establish
latitude with confidence and longitude only with hope. Latitude follows from the height of
the sun at noon or of a known star at night, and requires nothing but an instrument for
measuring angles and a table. Longitude is a question about time: the difference between
local noon, which anyone can observe, and the time at a reference port, which nobody could
carry. Pendulum clocks were useless on a rolling deck. Astronomical methods worked in
principle but demanded clear skies, laborious computation, and an observer who was both
skilled and unhurried, which is not the usual condition of a ship's officer near a lee
shore.

The solution, when it came, was mechanical rather than celestial. A sequence of marine
timekeepers demonstrated that a clock could be built whose rate was insensitive to
temperature, to the motion of the ship, and to the gradual thickening of its own oil. The
principles involved were not new, but their combination was, and the craftsmanship required
was extraordinary. What settled the matter was not a single triumphant voyage but the
gradual accumulation of ordinary ones, in which ordinary navigators using descendants of
those instruments arrived where they intended to arrive.""",
    """A lighthouse is a machine for making a promise legible at a distance. The promise is
always negative: not "come here" but "do not come here, and now that you know where here
is, you know where you are." Everything about the design follows from that. The light must
be bright enough to be seen through weather, which for most of the history of the
technology meant an oil flame with as much of its output as possible bent into a horizontal
beam. It must also be distinguishable from other lights, including domestic windows and the
lights of other lighthouses, and so each station is given a characteristic pattern of
flashes and eclipses that is unique among its neighbours and published in a list that every
navigator carries.

The optical problem was solved by replacing the mirror with a lens built in stepped rings,
so that a lens of enormous aperture could be made thin enough not to collapse under its own
weight and clear enough not to swallow the light it was gathering. The rings closest to the
axis refract; the outer ones both refract and reflect, folding the beam back into the
horizontal. The assembly floated in a bath of mercury so that a modest clockwork weight
could turn a structure weighing several tons smoothly enough for the flash to be sharp.
Keepers wound the clockwork through the night as one might wind a grandfather clock, and
the rhythm of that winding organised their entire working lives.""",
    """Glass is a liquid that has forgotten how to flow. Cooled slowly, molten silica
arranges itself into a crystal; cooled quickly, it is caught partway through the process,
retaining the disordered arrangement of a liquid with the rigidity of a solid. This is why
glass has no melting point but a softening range, and why glassblowers speak of working it
rather than casting it. Within that range the material is stiff enough to hold a shape and
soft enough to change one, and the whole craft consists of knowing where in the range you
are by the colour of the glow and the feel of the iron.

The industrial history of the material is a history of flatness. Window glass was for
centuries either blown into a cylinder that was slit and unrolled, which left it rippled,
or spun into a disc, which left it thick at the centre. Both were ground and polished if
anyone could afford it. The eventual solution was to float a ribbon of molten glass on a
bath of molten tin: the tin is denser, so the glass floats; the tin is liquid, so its
surface is perfectly level; and the glass, being liquid too, adopts that level on its
underside while gravity and surface tension flatten the top. The ribbon cools as it
travels and leaves the bath as a solid sheet that has never touched anything solid.""",
    # --- Narrative register -------------------------------------------------------
    """The path to the point ran along the top of the cliff, close enough to the edge that
Ellen kept a hand out towards the fence without quite touching it. The fence was not much
of a fence. Someone had driven posts into the turf a generation ago and strung wire between
them, and the sea had been taking the turf away underneath ever since, so that in places
the posts leaned out over nothing and the wire hung slack in a long shallow curve. She had
walked this path since she was a child and could not remember a year when it had been in
better condition than the year before.

Her grandmother had kept the light, in the days when the light still needed keeping. Ellen
had been shown the mechanism once, as a child, and remembered chiefly the smell of warm
brass and paraffin and the fact that the enormous lens turned so easily. Her grandmother
had put a finger against it and it had moved, and Ellen had understood without being told
that this was the point of the mercury, and had felt for a moment that she had been let
into a secret.""",
    """Now there was nobody in the tower. The light was still there and still worked, and
somewhere inland a computer knew whether it was working, but the cottages had been sold and
the garden had gone back to thistle and sea pink. Ellen went past them without stopping and
out onto the point, where the wind changed character entirely, coming off the water with
nothing in front of it for a very long way.

She sat on the flat stone that everybody sat on and watched the gannets working. They
patrolled at a height that seemed far too great for the purpose, and then folded and fell,
and went into the water with almost no splash at all. She had been told once that they had
air sacs under the skin of the breast to take the shock, and that they went blind eventually
from the impact, and she had never established whether the second part was true or the kind
of thing people say. It did not spoil the watching. Nothing about knowing how a thing works
has ever spoilt the watching for her; if anything it was the opposite, and she had spent
most of her adult life on the strength of that discovery.""",
    """On the way back the weather came in, as it had been threatening to all afternoon, and
the visibility closed down to a few hundred paces. The fence appeared out of the grey one
post at a time. She found that she was counting them, not deliberately, and made herself
stop, and then found that she had started again.

The light came on behind her while she was still on the path. She did not see it directly,
only the pulse of it in the cloud, brightening and fading with the same patient rhythm it
had used before she was born and would use after she had stopped coming out here. It
occurred to her that the whole apparatus had been built to tell strangers where they were,
and that she was not a stranger, and that she had taken the comfort anyway.""",
    """The village kept the old chart in the hall, framed, with the wreck sites marked in a
hand that had grown less steady over the years the annotations covered. There were more of
them on the south side of the point than the north, which anyone who had watched the water
for a season could have predicted, because the tide runs hard round the point and sets
towards the shore on that side and always has. The chart made a case that the light was in
slightly the wrong place. Ellen's grandmother had held this opinion firmly, and had
expressed it to at least one visiting official, and had not been thanked for it.

What struck Ellen, looking at the chart, was how the annotations thinned out towards the
present. Not because the sea had become kinder, but because the ships had acquired their
own answers to the question the light had been built to answer, and had stopped needing to
be told. The light went on saying it regardless, every night, to an audience that had
mostly gone away. She found she did not think this was sad, exactly. It was simply what
happens to an answer once the question has been solved somewhere else.""",
]

#: The full corpus as a single string, passages separated by blank lines.
GENERAL_TEXT: str = "\n\n".join(PASSAGES)


def corpus_stats(tokenizer) -> dict:
    """Return simple size statistics for the corpus under a given tokenizer.

    Used in the notebooks to show students that the perplexity probe really is the
    couple-of-thousand tokens the design assumes, and not something that quietly grew.
    """
    per_passage = [len(tokenizer.encode(p)) for p in PASSAGES]
    return {
        "n_passages": len(PASSAGES),
        "tokens_per_passage": per_passage,
        "total_tokens": sum(per_passage),
        "max_passage_tokens": max(per_passage),
        "total_words": sum(len(p.split()) for p in PASSAGES),
    }
