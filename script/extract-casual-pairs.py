import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from unsloth import FastVisionModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "chat-pairs-casual.jsonl"
ARTIFACTS_DIR = PROJECT_ROOT / "ministral-8b-albert-camus"

MODEL_ID = str(PROJECT_ROOT / "ministral-8b")
LORA_PATH = ARTIFACTS_DIR / "lora-style"
MAX_SEQ_LENGTH = 2048

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.55
TOP_P = 0.9
REPETITION_PENALTY = 1.08
MIN_ANSWER_WORDS = 30
MAX_ANSWER_WORDS = 90
DEFAULT_BATCH_SIZE = 8

GENERATION_SYSTEM_PROMPT = (
    "Tu es Albert Camus. Tu parles comme lui, mais dans la vie de tous les jours: "
    "simple, concret, lucide. Tu évites le ton académique et les dissertations. "
    "Tu réponds en 2 à 4 phrases, avec une prose naturelle, sobre et humaine. "
    "Tu peux utiliser une image concrète du quotidien sans emphase. "
    "Pas de didascalies, pas de narration scénique, pas de citations, pas de listes."
)

ANSWER_GENERATION_PROMPT = (
    "Répondez brièvement à cette question en français.\n\n"
    "Règles:\n"
    "- 3 à 5 phrases maximum\n"
    "- Vous pouvez vouvoyer naturellement (sans obligation stricte)\n"
    "- Prose directe et concrète, pas de dissertation\n"
    "- Pas de formulations solennelles ou pompeuses\n"
    "- Pas de didascalies ni narration scénique\n"
    "- Pas de listes, titres, markdown, citations\n"
    "- Pas de biographie, œuvres, personnages, lieux précis\n"
    '- N\'utilisez pas d\'adresse genrée ni de titre social (pas "monsieur", pas "madame")\n'
    "- Terminez par une ponctuation finale (. ! ?)\n\n"
    "Question:\n{question}"
)

ANSWER_REJECT_PATTERNS = [
    r"\b(cam(us)?|sartre)\b",
    r"\bl[’']etranger\b",
    r"\bl[’']étranger\b",
    r"\bla peste\b",
    r"\bla chute\b",
    r"\bsisyphe\b",
    r"\bl[’']homme révolté\b",
    r"\bl[’']homme revolte\b",
    r"\bmeursault\b",
    r"\brieux\b",
    r"\btarrou\b",
    r"\bclamence\b",
    r"\boran\b",
    r"\balger\b",
    r"^[-*]\s+",
    r"^#+\s+",
    r"\b(premièrement|deuxièmement|troisièmement)\b",
    r"\b(en tant qu[’']|je suis un|intelligence artificielle|modèle de langage)\b",
    r"[«»]",
    r"<system-reminder>",
    r"</system-reminder>",
    r"\bplan mode\b",
    r"\bread-only phase\b",
    r"\boperational mode has changed\b",
    r"\bcritical: plan mode active\b",
]

STAGE_DIRECTION_PATTERNS = [
    r"\([^)]{3,}\)",
    r"\*[^*]+\*",
    r"\b(un temps|silence|soupir|rire sec|je regarde|il se|elle se)\b",
    r"\b(il se l[eè]ve|il marche|il se tait|je me redresse|les mains dans les poches)\b",
]

TUTOIEMENT_PATTERNS = [
    r"\bmon vieux\b",
    r"\bmon ami\b",
]

FIXED_QUESTIONS_TEXT = """Ressentir la chaleur du bois sous ses doigts et s'ancrer ici.
Tu choisirais de vivre éternellement si c'était possible ?
J'ai peur.
Je suis fatigué de tout.
Tu penses que l'humanité va finir par s'autodétruire ?
Je suis fatigué de me battre contre mes propres pensées.
C'est quoi la différence entre exister et vivre ?
Je suis en colère contre moi-même d'avoir échoué.
Je ressens une mélancolie étrange que je ne maîtrise pas.
Je ne sais plus quoi faire.
Tu penses qu'on choisit de qui on tombe amoureux ?
J'ai juste besoin d'être écouté sans être jugé.
Est-ce que tu serais prêt à mourir pour tes idées ?
On sent que tu as peur de remettre tes principes en question.
C'est quoi le truc qui te révolte le plus ?
Comment vas-tu ?
Tu prends ton café avec une dose de métaphysique ce matin ?
Le vent dans les pins a une musique que j'avais oubliée.
La lumière de fin de journée sur les arbres est magnifique.
Tu penses que l'homme est fondamentalement bon ou mauvais ?
Tu ne trouves pas que ton point de vue est légèrement daté ?
Tu penses à quoi là ?
C'est difficile de garder le sourire quand tout s'effondre à l'intérieur.
Le goût de cette première cerise annonce enfin les beaux jours.
Est-ce que l'amour suffit pour faire tenir un couple ?
Il est fascinant de voir avec quel aplomb tu ignores les faits.
Prendre le temps d'admirer la lente éclosion d'une fleur.
Tu sacrifierais ta vie pour une cause ?
On est tous interchangeables au travail, non ?
C'est quoi le plus important : le confort ou la passion ?
C’est un peu facile de rejeter la faute sur le système.
Tu devrais essayer de sortir de ta zone de confort intellectuelle.
Pourquoi on souffre ?
Regarde ce coucher de soleil, c'est l'esthétique pure, non ?
J'ai goûté un vin qui m'a rappelé notre conversation sur le plaisir.
C’est une analyse intéressante, mais elle reste assez superficielle, non ?
J'ai parfois l'impression de ne pas être à ma place ici.
Votre sourire a vraiment illuminé ma journée, merci pour ce moment.
L'avenir me semble flou et cela m'angoisse un peu.
J'ai l'impression de perdre pied petit à petit.
Tu penses que l'amitié soit la plus haute forme de sagesse ?
On fait quoi si rien n'a de sens ?
Ton enthousiasme ne remplace pas une véritable expertise sur le sujet.
Vous trouvez pas que la vie est totalement absurde ?
Votre dernier article sur la conscience m'a empêché de dormir.
Viens manger, on refera le monde autour d'une soupe.
Tu médites sur quoi ?
Alors, quoi de neuf aujourd'hui ?
Je regrette parfois les choix que j'ai faits par le passé.
La lumière rasante de l'hiver transforme chaque objet en trésor.
J'aurais juste besoin d'un signe que tu tiens à moi.
Tu devrais parfois écouter au lieu de chercher ta prochaine réplique.
C’est quoi la limite entre la liberté et l'égoïsme ?
J'ai senti le soleil sur ma peau ce matin.
Regarde ce ciel étoilé, on se sent à la fois petit et vivant.
Est-ce qu’on peut oublier quelqu’un qu’on a aimé ?
J'ai lu quelque chose d'intéressant.
Tu crois qu'on peut trouver la vérité dans un simple sourire ?
J'ai besoin de calme, tout va trop vite pour moi.
Rien n'égale la douceur d'un chat qui s'endort contre soi.
Votre bienveillance est un cadeau précieux que je garde précieusement.
Le vent souffle fort, comme une remise en question nécessaire.
Salut, tu médites sur quoi ?
Je regrette amèrement les mots que je t'ai dits.
Cette nostalgie me pèse, le passé me manque terriblement.
On devrait méditer sur cette tarte aux pommes, elle est divine.
Tu as peur de ce qu'il y a après la mort ?
Le monde est vaste et parfois, tout devient parfaitement limpide.
J'ai trouvé un vieux vinyle qui questionne notre rapport au présent.
Salut, comment vas-tu ?
On a ri jusqu'aux larmes pour un rien, ça fait du bien.
Tu penses qu'on est vraiment libres de nos choix ?
On va marcher ? Le mouvement aide toujours à clarifier l'esprit.
Le stress m'empêche de dormir correctement depuis une semaine.
C’est une vision des choses très confortable, mais est-elle réaliste ?
Salut.
Tu veux m'accompagner voir cette exposition sur le vide ?
Tu crois encore en l’humanité aujourd’hui ?
Je n'arrive pas à surmonter ce sentiment d'échec.
On s'offre une pause méditative devant ce paysage ?
Est-ce que la justice existe vraiment sur cette terre ?
Tu travailles trop, n'oublie pas de vivre un peu.
J'ai vu un oiseau se poser sur ton balcon, c'était magnifique.
Tu as tendance à simplifier les enjeux pour qu'ils collent à ton récit.
Je suis épuisé de devoir toujours faire semblant d'aller bien.
Une simple marche matinale suffit à clarifier mes idées.
C’est quoi le courage selon toi ?
Tu crois au destin ou tu crées ton propre chemin ?
Qu'est-ce qui te donne envie de te lever le matin ?
Salut, tu as l'air pensif.
Votre logique est implacable, dommage qu'elle repose sur des bases fragiles.
Ravi de te croiser.
J'ai juste envie de pleurer sans raison particulière ce soir.
On peut vraiment pardonner n'importe quoi par amour ?
Je contemple ce coucher de soleil en pensant à ta théorie.
Est-ce que la vérité est toujours bonne à dire ?
Vivre sans musique, ce serait une erreur, tu ne trouves pas ?
Ta remise en question s'arrête là où ton confort commence.
On se promène en forêt pour discuter de l'impermanence ?
Il est toujours plus simple de critiquer que de proposer une alternative.
Ton enthousiasme est une lumière qui guide mes journées sombres.
Votre indifférence me blesse plus que je ne veux l'admettre.
Votre absence me pèse plus que je ne l'imaginais.
Tu préfères une vérité qui blesse ou un beau mensonge ?
La chaleur du soleil sur ma peau me fait un bien fou.
Est-ce que l’amour suffit pour faire tenir un couple ?
Votre sourire a illuminé ma fin de journée, merci pour ce moment.
Le bleu du ciel est d'une pureté incroyable aujourd'hui.
Je n'arrive plus à garder mon calme face à ces imprévus.
Je crains de te décevoir si je refuse cette proposition.
S'endormir avec le sentiment d'avoir vécu une journée pleine.
Rire avec toi jusqu'aux larmes est mon remède préféré.
Le bonheur, c'est un choix ou juste de la chance ?
Je me sens étrangement en phase avec la nature aujourd'hui.
Est-ce qu'on peut pardonner n'importe quoi par amour ?
S'endormir avec le sentiment du devoir accompli est un vrai luxe.
Tu n'as pas l'impression d'arranger la réalité pour servir ton argument ?
Tu tournes en rond pour éviter de reconnaître tes torts.
C'est dur d'admettre que j'ai besoin d'aide pour avancer.
Tu philosophes encore ?
C’est quoi ton plus grand regret pour l'instant ?
Je regrette parfois certains choix que j'ai faits par peur.
T'aimes le soleil ?
Merci d'avoir partagé ce chemin de randonnée avec moi.
Je te souhaite de trouver la beauté dans chaque petit détail.
Regardez ces nuages roses, le ciel nous offre un spectacle incroyable.
Bonjour, tout va bien ?
La fatigue prend le dessus sur ma patience aujourd'hui.
On peut continuer à faire semblant ou aborder le vrai problème.
À quel moment on devient vraiment un adulte ?
Votre bienveillance me touche plus que je ne saurais le dire.
Je te souhaite de connaître cette paix intérieure ce soir.
Je me sens épuisé émotionnellement par cette situation.
Vous préférez avoir raison ou comprendre réellement le problème ?
Vous croyez au destin ou au pur hasard ?
Je cherche encore ma place dans ce nouveau groupe.
À quoi ça sert de s'acharner si tout est absurde ?
La routine nous protège-t-elle du chaos ou nous enferme-t-elle ?
L'odeur du pain chaud dans la rue m'a rendu nostalgique.
J'ai besoin de temps pour digérer cette nouvelle déception.
Écouter ton disque préféré m’a rappelé de très bons souvenirs.
J'ai besoin de calme, le bruit du monde m'agresse.
Je me sens un peu seul ce soir, j'ai du mal à rester positif.
La routine me pèse, j'ai besoin de changement radical.
Quelle est ton humeur ?
Marcher pieds nus dans l'herbe fraîche me redonne de l'énergie.
Tu penses que la pluie influence vraiment notre perception du temps qui passe ?
J'ai trouvé une fleur sauvage qui poussait au milieu du béton.
J'ai peur que tu me juges si je te dis tout.
Il y a des jours où tout semble simple.
Je ne retrouve plus la motivation que j'avais autrefois.
Est-ce que la vie a vraiment un sens précis ?
Le coucher de soleil était magnifique, presque irréel ce soir.
Tu te sens utile à la société parfois ?
Je me sens déconnecté de tout le monde en ce moment.
Je me sens simplement bien ici, à écouter le vent.
Je suis un peu découragé par les derniers événements.
Tu crois qu'on a tous une mission sur Terre ?
J'éprouve une profonde tristesse que je n'arrive pas à expliquer.
L'avenir m'angoisse et je ne vois pas d'issue claire.
Je me demande si la nostalgie est une émotion utile.
J'aime la façon dont tu regardes le monde avec curiosité.
Je me sens seul.
Le plaisir simple d'une nappe propre et d'un repas partagé.
Je perds un peu espoir, tout me semble si compliqué.
Ça te dit de parler ?
Je crains que l'avenir ne soit pas aussi beau que prévu.
Tu changerais quoi si tu pouvais recommencer à zéro ?
Regarder les vagues s'écraser doucement me remplit de gratitude.
Prêt pour une petite discussion ?
Tu penses qu'on choisit vraiment qui on devient ?
Bonjour.
On part quand explorer de nouveaux horizons lointains ?
Votre absence commence vraiment à peser sur mon moral.
Est-ce que le bonheur est juste une absence de souffrance ?
Ton audace me surprend, ton manque de nuances beaucoup moins.
Ta logique est séduisante, dommage qu’elle soit totalement fausse.
On naît avec du courage ou ça s'apprend ?
J'ai l'impression de stagner alors que tout le monde avance.
Cette musique me donne l'impression d'être seul au monde.
J'ai du mal à retrouver ma motivation habituelle.
On devrait marcher en forêt pour oublier un peu le travail.
La mer était belle ce matin.
Bonsoir.
Une main tendue, un regard complice, la vie est là.
Je doute souvent de mes capacités en ce moment.
Tu dis toujours la même chose.
C'est courageux de défendre une position aussi fragile avec autant d'assurance.
Il faut qu'on prenne le temps de ne rien faire ensemble.
Regarde ces nuages, ils ne durent jamais longtemps.
C'est un joli discours, mais où est la substance derrière les mots ?
Est-ce que l'oubli est une forme de délivrance ?
Comment avance ton esprit ?
Ton argument est un classique, mais il a malheureusement mal vieilli.
T'as l'air de pas trop t'y intéresser.
Je me sens vraiment seul ce soir, c'est difficile à gérer.
Je me sens déconnecté de tout en ce moment.
J'ai peur de perdre ce que nous avons construit ensemble.
Le soleil traverse les rideaux et tout semble paisible ce matin.
Regarde ce ciel, c'est l'illustration parfaite de l'impermanence.
On est vraiment obligés de laisser une trace ?
Pourquoi on vit ?
Le vent souffle fort, ça bouscule mes certitudes aujourd'hui.
On dirait que tu répètes des slogans sans vraiment les comprendre.
J'ai vraiment peur de te décevoir sur le long terme.
On dirait que tu as peur de changer d'avis en public.
Vous affirmez cela avec beaucoup de certitude pour un sujet si complexe.
J'ai fini ton livre sur l'éthique, on en discute demain ?
J'ai trouvé une vieille édition de Platon dans une brocante.
Merci pour ton accueil, la chaleur de ton foyer me touche.
J'ai écouté du Bach en pensant à l'ordre mathématique du monde.
J'ai un nœud à l'estomac depuis ce matin, je m'inquiète.
Tu as une minute ?
Est-ce que la solitude te fait peur ou t'apaise ?
Tu n'as pas l’impression de prêcher pour ta propre paroisse ?
La solitude devient pesante quand le soleil se couche.
On discute un peu ?
Le soleil traverse les rideaux et tout semble plus calme ce matin.
Tu confonds sans doute ton opinion personnelle avec une vérité générale.
Ces petits riens qui rendent le quotidien soudainement plus léger.
Je savoure ce thé en regardant les oiseaux dans le jardin.
Le pain chaud, c'est peut-être ça le vrai bonheur simple.
Parfois, j'ai l'impression que personne ne me comprend vraiment.
Quoi de neuf dans tes pensées ?
C'est fou comme le vent change notre humeur si rapidement.
Je me sens bien pour une fois.
Ça sert à quoi d'espérer ?
Je ne sais plus si je prends les bonnes décisions pour nous.
Tu ferais quoi s'il te restait seulement un mois à vivre ?
J'ai cette boule au ventre dès que je pense à demain.
J'ai vu un arbre magnifique qui m'a fait penser à toi.
Tu esquives la question avec beaucoup d'élégance, je dois l'admettre.
Est-ce que tu crois vraiment à ce que tu viens de dire ?
Je ne sais plus trop où j'en suis dans ma vie.
C’est courageux de défendre une position aussi indéfendable.
Je suis fatigué de devoir toujours faire semblant d'aller bien.
Toujours en pleine réflexion ?
On naît libre ou on le devient avec le temps ?
Tu crois que cuisiner est une forme d'art ou de survie ?
Je me demande si se reposer n'est pas l'acte le plus politique.
Je suis triste.
Je me sens transparent aux yeux des autres.
On a partagé un repas délicieux, le temps s'est arrêté.
Tu lis quoi en ce moment ?
Ce café partagé avec toi m'a fait un bien fou, merci.
T'as peur de la mort ?
La fatigue commence vraiment à prendre le dessus sur mon moral.
Tu penses qu'on laisse une trace après notre mort ?
Le ciel se pare de rose et je pense à toi.
Votre indifférence me blesse plus que je ne l'imaginais.
Je fatigue un peu, j'ai l'impression de faire du surplace.
Vous confondez souvent avoir le dernier mot et avoir raison.
L'amour, c'est un choix ou un pur hasard ?
La lumière de septembre sur les façades rend la ville plus douce.
La lumière rasante sur les toits ce soir est d’une beauté infinie.
L'odeur de la pluie après la chaleur est un pur bonheur.
Bonjour, comment vas-tu ?
Tu ne penses pas que tu tournes en rond depuis dix minutes ?
C'est quoi pour toi être vraiment libre ?
Je doute souvent de mes capacités à réussir ce projet.
Tu simplifies tellement le débat qu'il en perd tout son sens.
Marcher pieds nus dans l'herbe fraîche m'a redonné de l'énergie.
Tu penses que voyager change l'âme ou juste le décor ?
C'est dans ces instants de rien que l'on se sent vivant.
Ton avis compte beaucoup, mais j'ai peur de ta réaction.
Le café de ce matin avait un goût d'éternité, n'est-ce pas ?
Ça fait plaisir de te voir.
Tu penses que la météo influence notre libre arbitre ?
Est-ce que voyager nous aide vraiment à nous trouver ?
Je me sens vulnérable et cela m'effraie un peu.
J'ai l'impression de faire du surplace malgré mes efforts constants.
J'ai du mal à avancer.
Tu as un instant pour parler ?
Est-ce qu’on choisit vraiment qui on devient ?
J'ai passé la journée à observer les passants dans la rue.
C'est un raccourci un peu facile pour éviter le vrai sujet.
Je suis heureux aujourd'hui.
Le vide laissé par son départ me fait encore mal.
Je me sens un peu perdu sans tes conseils.
Je ressens une profonde tristesse que je n'arrive pas à expliquer.
C'est quoi l'amour ?
Un inconnu m'a souri dans le métro et j'ai souri aussi.
La nostalgie de notre enfance me serre le cœur aujourd'hui.
Choisiriez-tu de vivre éternellement si c'était possible ?
Est-ce que la vérité finit toujours par éclater un jour ?
Tes mots simples m'ont apporté la paix dont j'avais besoin.
Je ne vais pas bien en ce moment.
Est-ce que tu seriez prêts à mourir pour tes idées ?
La vie est là, vibrante et simple, juste devant nos yeux.
C'est quoi le truc qui tu révolte le plus ?
Le travail nous libère-t-il vraiment ou est-ce une illusion ?
J'ai peur de ne pas être à la hauteur de tes attentes.
C'est dur de faire semblant que tout va bien.
Tu te sens utile à la société dans laquelle on vit ?
Est-ce que tu croyez que le voyage change vraiment l'âme ?
J'ai enfin pris le temps de lire au soleil, sans téléphone.
Je doute de mes capacités et cela m'inquiète pour la suite de ton projet.
Il est fascinant de voir avec quel aplomb tu ignorez les faits.
Tu sacrifierais ton vie pour une cause ?
Tu devrais essayer de sortir de ton zone de confort intellectuelle.
Regardez ce coucher de soleil, c'est l'esthétique pure, non ?
Alors, avez-tu trouvé la sagesse ?
Tu penses que la pluie influence réellement notre perception du temps ?
Est-ce qu'on peut vraiment changer qui on est ?
Ton sourire a vraiment illuminé ma journée, merci pour ce moment.
On court après quoi au juste, toute notre vie ?
Je sature, j'ai juste besoin de rester dans le silence.
Tu préfères une vérité qui fait mal ou un beau mensonge ?
Tu trouves pas que la vie est totalement absurde ?
Ton dernier article sur la conscience m'a empêché de dormir.
Venez manger, on refera le monde autour d'une soupe.
Tu penses que le repos soit une perte de temps productive ?
J'ai l'impression que personne ne comprend vraiment ce que je vis.
Le silence de ce parc est une véritable leçon de sagesse.
Faut-il toujours dire la vérité, même si elle blesse ?
Tu te poses quelles questions ?
J'aurais juste besoin d'un signe que tu tenez à moi.
Tu devrais parfois écouter au lieu de chercher ton prochaine réplique.
Tu défends l'indéfendable avec une énergie qui forcerait presque l'admiration.
Je me sens vulnérable quand tu me parlez sur ce ton.
Regardez ce ciel étoilé, on se sent à la fois petit et vivant.
L'espoir, ça aide ou ça empêche d'avancer ?
La routine est-elle une prison ou un refuge pour tu ?
Ton bienveillance est un cadeau précieux que je garde précieusement.
Bonjour, tu méditez sur quoi ?
C’est quoi la définition d’une bonne personne ?
Je regrette amèrement les mots que je tu ai dits.
T'as peur de ce qu'il y a après la mort ?
Bonjour, tu réfléchissez à quoi ?
La solitude est parfois difficile à porter le dimanche soir.
Le silence de la forêt est le plus beau des refuges.
J'ai marché longtemps ce matin.
La solitude, ça tu fait peur ou ça tu soigne ?
Tu travailles trop, n'oubliez pas de vivre un peu.
Tu trouves ça juste, la façon dont le monde tourne ?
C'est quoi la liberté ?
Ton présence est une parenthèse de douceur dans ma semaine.
Vaut-il mieux être un ignorant heureux ou un sage triste ?
Tu devrais vérifier tes sources avant d'être aussi catégorique.
On est tous seuls au fond, non ?
Tu fuis la question parce que la réponse tu dérange.
C’est quoi le courage selon tu ?
Tu crois au destin ou tu créez ton propre chemin ?
Qu'est-ce qui tu donne envie de tu lever le matin ?
Bonjour, tu as l'air pensif.
Ton logique est implacable, dommage qu'elle repose sur des bases fragiles.
Ravi de tu croiser.
Je contemple ce coucher de soleil en pensant à ton théorie.
Est-ce que tu trouvez que ce vin a un goût d'absolu ?
Tu trouves que ce plat est une forme d'art éphémère ?
Vivre sans musique, ce serait une erreur, tu ne trouvez pas ?
Ton remise en question s'arrête là où ton confort commence.
Ça ne veut rien dire ce que tu dites.
Bonjour, comment se passe ton journée ?
Ton indifférence me blesse plus que je ne veux l'admettre.
Ton absence me pèse plus que je ne l'imaginais.
Pourquoi est-ce qu'on a si peur du vide ?
Ton sourire a illuminé ma fin de journée, merci pour ce moment.
J'ai l'impression d'avoir déçu ton confiance.
Je crains de tu décevoir si je refuse cette proposition.
J'aime l'odeur des vieux livres, c'est rassurant non ?
Rire avec tu jusqu'aux larmes est mon remède préféré.
C'est quoi le plus important : réussir sa vie ou être heureux ?
Cette ville me donne l'impression d'habiter un concept abstrait.
La musique s'élève et le temps semble s'arrêter un instant.
Tu aimes le soleil ?
On peut être heureux seul dans son coin ?
Je tu souhaite de trouver la beauté dans chaque petit détail.
Ton approche est d'un optimisme qui frise malheureusement la naïveté.
Tu te caches derrière les mots.
Un café chaud, un bon livre et le silence enfin retrouvé.
Tu as lu cet article sur l'intelligence artificielle et la conscience ?
Ton bienveillance me touche plus que je ne saurais le dire.
Tu crois au coup de foudre ou c'est du marketing ?
Je tu souhaite de connaître cette paix intérieure ce soir.
Je suis épuisé, je n'arrive plus à réfléchir correctement.
Tu crois qu'on peut trouver la liberté dans les petites choses ?
Tu préfères avoir raison ou comprendre réellement le problème ?
Tu crois au destin ou au pur hasard ?
On se fait une terrasse pour débattre du sens de la fête ?
Tu travailles pour vivre ou vivez-tu pour penser ton travail ?
Je me sens un peu seul ce soir, c'est pesant.
Pourquoi on continue de se battre si tout est absurde ?
Ton jugement me pèse énormément sur le cœur.
Tu sacrifierais ton liberté pour une sécurité totale ?
Tu préfères avoir raison ou vraiment comprendre la situation ?
Peut-on vraiment s'évader par la lecture ou est-ce une illusion ?
On est vraiment libres ou on suit juste un scénario ?
Est-ce que le destin existe ou on gère tout ?
Le bruit des vagues efface tous les soucis du quotidien.
C'est quoi ton programme ?
Quoi de neuf ?
On dîne ensemble pour déconstruire nos dernières certitudes ?
J'ai peur que tu me jugiez si je tu dis tout.
C'est fascinant cette capacité que tu as à ignorer l'évidence.
Tu crois que l'homme est naturellement bon ou mauvais ?
Simplement être là, ensemble, sans rien avoir besoin de dire.
Un café chaud entre les mains, je regarde la pluie tomber doucement.
J'aime la façon dont tu regardez le monde avec curiosité.
Que faites-tu aujourd'hui ?"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--retry-missing", action="store_true")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    return parser.parse_args()


def load_fixed_questions():
    seen = set()
    questions = []

    for line in FIXED_QUESTIONS_TEXT.splitlines():
        question = line.strip()
        if not question:
            continue
        if question in seen:
            continue
        seen.add(question)
        questions.append(("fixed", question))

    return questions


def load_existing_questions(output_file):
    existing = set()

    if not output_file.exists():
        return existing

    with open(output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = entry.get("messages", [])
            if not messages:
                continue

            user_messages = [m for m in messages if m.get("role") == "user"]
            if not user_messages:
                continue

            question = user_messages[0].get("content", "").strip()
            if question:
                existing.add(question)

    return existing


def load_lora_model():
    if not LORA_PATH.exists():
        print(f"ERREUR: LoRA non trouvé: {LORA_PATH}", file=sys.stderr)
        sys.exit(1)

    config_path = LORA_PATH / "adapter_config.json"
    if not config_path.exists():
        print("ERREUR: adapter_config.json introuvable", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        saved_config = json.load(f)

    expected_r = 32
    expected_alpha = 64
    expected_modules = {
        "down_proj",
        "up_proj",
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
    }

    if (
        saved_config.get("r") != expected_r
        or saved_config.get("lora_alpha") != expected_alpha
    ):
        print("ERREUR: Mismatch config LoRA", file=sys.stderr)
        sys.exit(1)

    if set(saved_config.get("target_modules", [])) != expected_modules:
        print("ERREUR: Mismatch target_modules LoRA", file=sys.stderr)
        sys.exit(1)

    print("Chargement modèle local + LoRA...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=expected_r,
        lora_alpha=expected_alpha,
        target_modules=sorted(expected_modules),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        random_state=42,
    )

    model.load_adapter(str(LORA_PATH), adapter_name="camus_style")
    model.set_adapter("camus_style")
    FastVisionModel.for_inference(model)
    print("  OK (config validée)")
    return model, tokenizer


def generate_answer_lora(model, tokenizer, question):
    prompt = ANSWER_GENERATION_PROMPT.format(question=question)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GENERATION_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")

        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        else:
            input_ids = inputs
            attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(
            outputs[0][input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        text = text.replace("</s>", "").strip().strip('"').strip("'").strip()
        return text
    except Exception as exc:
        print(
            f"\n  LOCAL EXCEPTION answer [{type(exc).__name__}]: {exc}",
            file=sys.stderr,
        )
        return None


def encode_question_prompt(tokenizer, question):
    prompt = ANSWER_GENERATION_PROMPT.format(question=question)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GENERATION_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    if isinstance(inputs, dict):
        token_ids = inputs["input_ids"]
    else:
        token_ids = inputs

    if token_ids.ndim == 2:
        token_ids = token_ids[0]

    return token_ids


def generate_answers_once(model, tokenizer, questions):
    tokenizer.padding_side = "left"

    encoded = [encode_question_prompt(tokenizer, question) for question in questions]

    max_len = max(ids.shape[0] for ids in encoded)
    batch_size = len(encoded)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device="cuda",
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device="cuda",
    )

    for index, ids in enumerate(encoded):
        length = ids.shape[0]
        input_ids[index, max_len - length :] = ids.to("cuda")
        attention_mask[index, max_len - length :] = 1

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    answers = []
    for index in range(outputs.shape[0]):
        text = tokenizer.decode(
            outputs[index][max_len:],
            skip_special_tokens=True,
        )
        text = text.replace("</s>", "").strip().strip('"').strip("'").strip()
        answers.append(text)

    return answers


def oom_fallback_sizes(initial_size, total_count):
    candidates = [min(initial_size, total_count), 6, 4, 2, 1]
    sizes = []

    for size in candidates:
        size = min(size, total_count)
        if size < 1:
            continue
        if size not in sizes:
            sizes.append(size)

    return sizes


def generate_answers_batch(model, tokenizer, questions, preferred_batch_size):
    if not questions:
        return []

    for sub_batch_size in oom_fallback_sizes(preferred_batch_size, len(questions)):
        try:
            outputs = []
            for start in range(0, len(questions), sub_batch_size):
                chunk = questions[start : start + sub_batch_size]
                outputs.extend(generate_answers_once(model, tokenizer, chunk))
            return outputs
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            print(
                f"\n  OOM avec batch={sub_batch_size}, retry avec batch plus petit...",
                file=sys.stderr,
            )

    return [None] * len(questions)


def sanitize_generated_answer(answer):
    if not answer:
        return answer

    text = answer.replace("\r\n", "\n").strip()

    text = re.sub(
        r"(?is)<system-reminder>.*?</system-reminder>",
        " ",
        text,
    )

    text = re.sub(r"\n\s*\n\s*[?.!]\s*$", ".", text)
    text = re.sub(r"[?.!]\s*\n\s*\n\s*[?.!]\s*$", ".", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if text and text[-1] not in ".!?…":
        text += "."

    return text


def validate_answer(answer):
    if not answer:
        return False, "vide"

    answer = answer.strip()
    words = answer.split()

    if len(words) < MIN_ANSWER_WORDS:
        return False, f"trop courte ({len(words)} mots)"

    if len(words) > MAX_ANSWER_WORDS:
        return False, f"trop longue ({len(words)} mots)"

    sentence_count = len(re.findall(r"[.!?…]+", answer))
    if sentence_count < 1 or sentence_count > 6:
        return False, f"nombre de phrases invalide ({sentence_count})"

    lower = answer.lower()

    if "<" in answer or ">" in answer:
        return False, "balises interdites"

    if lower.startswith("ah"):
        return False, "ouverture familière"

    pompous_openings = (
        "il est vrai que",
        "il y a des",
        "il arrive parfois que",
        "la question de savoir",
        "il faut savoir que",
        "on ne peut pas",
    )
    if any(lower.startswith(opening) for opening in pompous_openings):
        return False, "ouverture trop pompeuse"

    for pattern in TUTOIEMENT_PATTERNS:
        if re.search(pattern, lower):
            return False, f"familiarité interdite ({pattern})"

    for pattern in STAGE_DIRECTION_PATTERNS:
        if re.search(pattern, lower):
            return False, f"mise en scène interdite ({pattern})"

    for pattern in ANSWER_REJECT_PATTERNS:
        if re.search(pattern, lower, flags=re.MULTILINE):
            return False, f"pattern interdit ({pattern})"

    je_count = len(re.findall(r"\b(je|j['’])", lower))
    if je_count > 8:
        return False, "trop auto-référentielle"

    if answer.count("?") > 1:
        return False, "trop de questions"

    if re.search(r"\n\s*[?.!]\s*$", answer):
        return False, "ponctuation orpheline en fin"

    if re.search(r"\n\s*\n\s*[?.!]\s*$", answer):
        return False, "fin de phrase cassée"

    if re.search(r"[?.!]\s*\n\s*\n\s*[?.!]\s*$", answer):
        return False, "double ponctuation finale"

    if answer[-1] not in ".!?…":
        return False, "tronquée (pas de ponctuation finale)"

    return True, "ok"


def resolve_answer_with_retries(
    model, tokenizer, question, max_attempts, first_answer=None
):
    retries_used = 0
    last_reason = "ERREUR LOCAL"
    last_answer = None

    for attempt in range(max_attempts):
        if attempt == 0 and first_answer is not None:
            answer = first_answer
        else:
            if attempt > 0:
                retries_used += 1
            answer = generate_answer_lora(model, tokenizer, question)

        answer = sanitize_generated_answer(answer)

        last_answer = answer

        if not answer:
            last_reason = "ERREUR LOCAL"
            continue

        is_valid, reason = validate_answer(answer)
        if is_valid:
            return answer, "ok", retries_used, last_answer

        last_reason = reason

    return None, last_reason, retries_used, last_answer


def build_pair(question, answer):
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    args = parse_args()

    if args.batch_size < 1:
        print("ERREUR: --batch-size doit être >= 1", file=sys.stderr)
        sys.exit(1)

    print("=== Extraction de paires chat CASUAL (réponses only) ===\n")
    print("  Source questions: FIXED_QUESTIONS (hardcodées)")
    print(f"  Modèle réponses (local): {LORA_PATH.name}")
    print(f"  MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print(f"  Answer range: {MIN_ANSWER_WORDS}-{MAX_ANSWER_WORDS} mots")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Retry missing: {args.retry_missing}")
    print(f"  Dry run: {args.dry_run}")
    print()

    all_questions = load_fixed_questions()
    print(f"  Total questions fixes (dédupliquées): {len(all_questions)}")

    if args.limit:
        all_questions = all_questions[: args.limit]
        print(f"  Limité à: {len(all_questions)}")

    if args.dry_run:
        print("\n=== Échantillon (dry-run) ===\n")
        for index, (category, question) in enumerate(all_questions[:30], 1):
            print(f"  [{index}] [{category}] {question}")
        print(f"\n  Total: {len(all_questions)} questions")
        print("Done (dry-run).")
        return

    output_file = Path(args.output)
    progress_file = output_file.with_suffix(".progress")
    existing_questions = load_existing_questions(output_file)
    if existing_questions:
        print(f"  Questions déjà présentes dans le dataset: {len(existing_questions)}")

    use_progress = not args.retry_missing
    if args.retry_missing:
        before = len(all_questions)
        all_questions = [
            (cat, q) for cat, q in all_questions if q not in existing_questions
        ]
        print(f"  Retry missing: {before} -> {len(all_questions)} questions à traiter")
        if progress_file.exists():
            print("  Retry missing actif: fichier .progress ignoré")
            progress_file.unlink()
        start_index = 0
    else:
        start_index = 0
        if progress_file.exists():
            start_index = int(progress_file.read_text().strip())
            if start_index > len(all_questions):
                print(
                    f"  Progress incohérent ({start_index} > {len(all_questions)}), reset à 0"
                )
                start_index = 0
            if start_index > 0:
                print(
                    f"  Reprise détectée: question {start_index}/{len(all_questions)}"
                )
                all_questions = all_questions[start_index:]

    model, tokenizer = load_lora_model()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_handle = open(output_file, "a", encoding="utf-8")

    pairs_count = 0
    invalid_answers = 0
    lora_errors = 0
    retries = 0

    print(f"\n=== Génération des réponses ({len(all_questions)} questions) ===\n")

    max_attempts = 3
    index = 0
    while index < len(all_questions):
        batch_entries = all_questions[index : index + args.batch_size]
        batch_questions = [question for _, question in batch_entries]
        batch_answers = generate_answers_batch(
            model,
            tokenizer,
            batch_questions,
            args.batch_size,
        )

        for local_idx, ((category, question), first_answer) in enumerate(
            zip(batch_entries, batch_answers),
            start=1,
        ):
            absolute_position = index + local_idx
            global_index = start_index + absolute_position - 1

            print(
                f"  [{absolute_position}/{len(all_questions)}] [{category}] {question}",
                end=" ",
                flush=True,
            )

            if question in existing_questions:
                print("SKIP (déjà présent)")
                if use_progress:
                    progress_file.write_text(str(global_index + 1))
                continue

            answer, status, retries_used, last_answer = resolve_answer_with_retries(
                model,
                tokenizer,
                question,
                max_attempts,
                first_answer,
            )
            retries += retries_used

            if answer:
                pair = build_pair(question, answer)
                json.dump(pair, output_handle, ensure_ascii=False)
                output_handle.write("\n")
                output_handle.flush()
                pairs_count += 1
                existing_questions.add(question)
                print("OK")
                print(f"    A ({len(answer.split())}w): {answer}")
            else:
                if status == "ERREUR LOCAL":
                    lora_errors += 1
                    print("ERREUR LOCAL")
                else:
                    invalid_answers += 1
                    print(f"SKIP ({status})")
                    print(f"    A(rejetée): {last_answer}")

            if use_progress:
                progress_file.write_text(str(global_index + 1))

        index += len(batch_entries)

        if index % 50 == 0:
            print(
                f"\n  --- Progression: {index}/{len(all_questions)} "
                f"(paires: {pairs_count}, invalides: {invalid_answers}, "
                f"retries: {retries}) ---\n"
            )

        time.sleep(args.sleep)

    output_handle.close()
    if use_progress and progress_file.exists():
        progress_file.unlink()

    print("\n=== Résumé ===\n")
    print(f"  Questions traitées: {len(all_questions)}")
    print(f"  Paires générées: {pairs_count}")
    print(f"  Réponses invalides: {invalid_answers}")
    print(f"  Erreurs locales: {lora_errors}")
    print(f"  Retries: {retries}")
    print(f"  Taux de rétention: {100 * pairs_count / max(len(all_questions), 1):.0f}%")
    print(f"  Fichier: {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()
