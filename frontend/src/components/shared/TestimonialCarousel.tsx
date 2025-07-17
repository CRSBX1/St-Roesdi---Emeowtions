import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Star, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const testimonials = [
  {
    name: "Sarah Johnson",
    text: "Emeowtions helped me understand why my cat Whiskers was acting stressed. The AI recommendations were spot-on!",
    rating: 5,
    location: "New York, NY"
  },
  {
    name: "Mike Chen", 
    text: "My Maine Coon's happiness levels have improved so much since using this app. Amazing insights!",
    rating: 5,
    location: "San Francisco, CA"
  },
  {
    name: "Emma Davis",
    text: "Finally, I can understand what my rescue cat is feeling. The personalized solutions are incredible.",
    rating: 5,
    location: "Austin, TX"
  },
  {
    name: "David Rodriguez",
    text: "As a veterinarian, I recommend Emeowtions to all my clients. It's like having a cat whisperer in your pocket!",
    rating: 5,
    location: "Miami, FL"
  },
  {
    name: "Jennifer Kim",
    text: "My shy Persian finally seems more confident after following the app's behavioral suggestions.",
    rating: 5,
    location: "Seattle, WA"
  },
  {
    name: "Robert Taylor",
    text: "The voice analysis is incredibly accurate. It detected my cat's anxiety before I even noticed it myself.",
    rating: 5,
    location: "Chicago, IL"
  },
  {
    name: "Lisa Wang",
    text: "Three cats, three different personalities - Emeowtions helps me understand each one perfectly.",
    rating: 5,
    location: "Los Angeles, CA"
  },
  {
    name: "Mark Thompson",
    text: "The premium features are worth every penny. My senior cat is happier than ever!",
    rating: 5,
    location: "Boston, MA"
  },
  {
    name: "Amanda Foster",
    text: "From aggressive to affectionate - this app helped transform my relationship with my rescue cat.",
    rating: 5,
    location: "Denver, CO"
  },
  {
    name: "Carlos Mendoza",
    text: "The AI recommendations helped me create the perfect environment for my anxious kitten.",
    rating: 5,
    location: "Phoenix, AZ"
  },
  {
    name: "Rachel Green",
    text: "I love how the app tracks emotional patterns over time. It's like a health diary for my cat's mind!",
    rating: 5,
    location: "Portland, OR"
  },
  {
    name: "Tommy Lee",
    text: "Professional cat groomer here - I use Emeowtions to better understand my feline clients. Game changer!",
    rating: 5,
    location: "Nashville, TN"
  },
  {
    name: "Sophie Miller",
    text: "The video analysis feature is fantastic. Now I can analyze my cat's behavior even when I'm at work.",
    rating: 5,
    location: "Atlanta, GA"
  }
];

const TestimonialCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  // Auto-rotate testimonials
  useEffect(() => {
    if (!isAutoPlaying) return;
    
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % testimonials.length);
    }, 4000);

    return () => clearInterval(interval);
  }, [isAutoPlaying]);

  const goToPrevious = () => {
    setIsAutoPlaying(false);
    setCurrentIndex((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  };

  const goToNext = () => {
    setIsAutoPlaying(false);
    setCurrentIndex((prev) => (prev + 1) % testimonials.length);
  };

  // Get 3 testimonials to display
  const getVisibleTestimonials = () => {
    const visible = [];
    for (let i = 0; i < 3; i++) {
      const index = (currentIndex + i) % testimonials.length;
      visible.push(testimonials[index]);
    }
    return visible;
  };

  return (
    <section className="py-16 px-4 bg-gradient-to-br from-secondary/30 to-lavender/30">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">What Cat Parents Say</h2>
          <p className="text-muted-foreground">Join thousands of happy cat owners using Emeowtions</p>
        </div>
        
        <div className="relative">
          <div className="grid md:grid-cols-3 gap-6">
            {getVisibleTestimonials().map((testimonial, index) => (
              <Card 
                key={`${currentIndex}-${index}`} 
                className="bg-card hover:shadow-lg transition-all duration-300 border-2 hover:border-primary/20"
              >
                <CardContent className="p-6">
                  <div className="flex mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                    ))}
                  </div>
                  <p className="text-card-foreground mb-4 min-h-[60px]">"{testimonial.text}"</p>
                  <div className="border-t pt-4">
                    <p className="font-semibold text-sm">{testimonial.name}</p>
                    <p className="text-xs text-muted-foreground">{testimonial.location}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Navigation buttons */}
          <Button
            variant="outline"
            size="icon"
            className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 bg-background/80 hover:bg-background"
            onClick={goToPrevious}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          
          <Button
            variant="outline"
            size="icon"
            className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 bg-background/80 hover:bg-background"
            onClick={goToNext}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {/* Dots indicator */}
        <div className="flex justify-center mt-8 space-x-2">
          {testimonials.map((_, index) => (
            <button
              key={index}
              className={`w-2 h-2 rounded-full transition-colors ${
                index === currentIndex ? 'bg-primary' : 'bg-muted'
              }`}
              onClick={() => {
                setCurrentIndex(index);
                setIsAutoPlaying(false);
              }}
            />
          ))}
        </div>

        <div className="text-center mt-6">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsAutoPlaying(!isAutoPlaying)}
            className="text-muted-foreground hover:text-foreground"
          >
            {isAutoPlaying ? "Pause" : "Resume"} Auto-play
          </Button>
        </div>
      </div>
    </section>
  );
};

export default TestimonialCarousel;