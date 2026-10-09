// SmartWaste AI frontend helpers
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("input[type=datetime-local]").forEach(input => {
    if (!input.value) {
      const now = new Date();
      now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
      input.value = now.toISOString().slice(0,16);
    }
  });
});
