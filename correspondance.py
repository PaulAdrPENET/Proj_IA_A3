import json

descr_motif_traj = {
  1:"Autres",
  2:"Courses - achats",
  3:"Domicile - école",
  4:"Domicile - travail",
  5:"Promenade - loisirs",
  6:"Utilisation professionnelle"
}
json_descr_motif_traj = json.dumps(descr_motif_traj)
print(json_descr_motif_traj)

descr_dispo_secu = {
  1:"Autre - Non déterminable ",
  2:"Autre - Non utilisé ",
  3:"Autre - Utilisé ",
  4:" Présence d un casque - Utilisation non déterminable",
  5:"Présence d un casque non utilisé",
  6:"Présence d un dispositif enfant non utilisé",
  7:"Présence d un équipement réfléchissant non utilisé",
  8:"Présence d une ceinture de sécurité - Utilisation non déterminable",
  9:"Présence de ceinture de sécurité non utilisée",
  10:"Présence dispositif enfant - Utilisation non déterminable",
  11:"Présence équipement réfléchissant - Utilisation non déterminable",
  12:"Utilisation d un casque",
  13:"Utilisation d un dispositif enfant",
  14:"Utilisation d un équipement réfléchissant",
  15:"Utilisation d une ceinture de sécurité"
}
json_descr_dispo_secu = json.dumps(descr_dispo_secu)

descr_athmo={
  1:"Autre",
  2:"Brouillard – fumée",
  3:"Neige – grêle",
  4:"Normale",
  5:"Pluie forte",
  6:"Pluie légère",
  7:"Temps couvert",
  8:"Temps éblouissant",
  9:"Vent fort – tempête"
}
json_descr_athmo = json.dumps(descr_athmo)

descr_grav={
  1:"Indemne",
  2:"Tué",
  3:"Blessé léger",
  4:"Blessé hospitalisé"
}
json_descr_grav = json.dumps(descr_grav)


descr_type_col={
  1:"Autre collision",
  2:"Deux véhicules - Frontale",
  3:"Deux véhicules – Par l’arrière",
  4:"Deux véhicules – Par le coté",
  5:"Sans collision",
  6:"Trois véhicules et plus – Collisions multiples",
  7:"Trois véhicules et plus – En chaîne"


}
json_descr_type_col = json.dumps(descr_type_col)

description_intersection={
  1:"Autre intersection",
  2:"Giratoire",
  3:"Hors intersection",
  4:"Intersection à plus de 4 branches",
  5:"Intersection en T",
  6:"Intersection en X",
  7:"Intersection en Y",
  8:"Passage à niveau",
  9:"Place"
}
json_description_intersection = json.dumps(description_intersection)

descr_etat_surf={
  1:"Autre",
  2:"Boue",
  3:"Corps gras – huile",
  4:"Enneigée",
  5:"Flaques",
  6:"Inondée",
  7:"Mouillée",
  8:"Normale",
  9:"Verglacée"
}
json_descr_etat_surf = json.dumps(descr_etat_surf)


descr_lum={
  1:"Crépuscule ou aube",
  2:"Nuit avec éclairage public allumé",
  3:"Nuit avec éclairage public non allumé",
  4:"Nuit sans éclairage public",
  5:"Plein jour"
}
json_descr_lum = json.dumps(descr_lum)

descr_agglo={
  1:"En agglomération",
  2:"Hors agglomération"
}
json_descr_agglo = json.dumps(descr_agglo)


descr_cat_veh={
  1:"Autobus",
  2:"Autocar",
  3:"Autre véhicule",
  4:"Bicyclette",
  5:"Cyclomoteur <50cm3",
  6:"Engin spécial",
  7:"Motocyclette > 125 cm3",
  8:"Motocyclette > 50 cm3 et <= 125 cm3",
  9:"PL > 3,5T + remorque",
  10:"PL seul > 7,5T",
  11:"PL seul 3,5T <PTCA <= 7,5T",
  12:"Quad léger <= 50 cm3 (Quadricycle à moteur non carrossé)",
  13:"Quad lourd > 50 cm3 (Quadricycle à moteur non carrossé)",
  14:"Scooter < 50 cm3",
  15:"Scooter > 125 cm3",
  16:"Scooter > 50 cm3 et <= 125 cm3",
  17:"Tracteur agricole",
  18:"Tracteur routier + semi-remorque",
  19:"Tracteur routier seul",
  20:"Train",
  21: "Tramway",
  22: "VL seul",
  23: "Voiturette (Quadricycle à moteur carrossé) (anciennement 'voiturette ou tricycle à moteur')",
  24: "VU seul 1,5T <= PTAC <= 3,5T avec ou sans remorque"
}
json_descr_cat_veh = json.dumps(descr_cat_veh)
