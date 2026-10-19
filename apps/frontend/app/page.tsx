import { redirect } from "next/navigation";

export default function Home() {
  // The Writing Studio is public; the platform console lives under /app (login required).
  redirect("/studio");
}
