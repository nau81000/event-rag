import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";

const resources = {
  en: {
    translation: {
      appTitle: "Event recommandation assistant",
      poweredBy: "Powered by Ollama · Qdrant · React · FastAPI",
      welcomeMessage: "Hi! What are you looking for? 😊",
      inputPlaceholder: "Looking for event...",
      searchResult: "No event found",
      connexionError: "Sorry — I couldn't join the server"
    },
  },
  fr: {
    translation: {
      appTitle: "Assistant pour la recommandation d'évènements",
      poweredBy: "Propulsé par Ollama · Qdrant · React · FastAPI",
      welcomeMessage: "Bonjour! Que recherchez-vous? 😊",
      inputPlaceholder: "Rechercher des évènements...",
      searchResult: "Aucun évènement trouvé",
      connexionError: "Désolé — Je n'ai pas pu joindre le serveur"
    },
  },
};

i18n
  .use(LanguageDetector) // auto-detect browser language
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: "en",
    supportedLngs: ["en", "fr"],
    interpolation: { escapeValue: false },
    detection: {
      order: ["localStorage", "navigator"],
      caches: ["localStorage"], // remember choice
    },
  });

export default i18n;